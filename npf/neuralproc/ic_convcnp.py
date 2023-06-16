"""Module for convolutional [conditional | latent] neural processes"""
import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from npf.architectures import CNN, ResConvBlock, SetConv, discard_ith_arg
from npf.utils.initialization import weights_init
from torch.distributions.independent import Independent

from .ic_base import ICNeuralProcessFamily
from .helpers import (
    collapse_z_samples_batch,
    pool_and_replicate_middle,
    replicate_z_samples,
)
from .np import LNP

logger = logging.getLogger(__name__)

__all__ = ["ICConvCNP"]


class ICConvCNP(ICNeuralProcessFamily):
    """
    Convolutional conditional neural process [1].

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    density_induced : int, optional
        Density of induced-inputs to use. The induced-inputs will be regularly sampled.

    Interpolator : callable or str, optional
        Callable to use to compute cntxt / trgt to and from the induced points.  {(x^k, y^k)}, {x^q} -> {y^q}.
        It should be constructed via `Interpolator(x_dim, in_dim, out_dim)`. Example:
            - `SetConv` : uses a set convolution as in the paper.
            - `"TransformerAttender"` : uses a cross attention layer.

    CNN : nn.Module, optional
        Convolutional model to use between induced points. It should be constructed via
        `CNN(r_dim)`. Important : the channel needs to be last dimension of input. Example:
            - `partial(CNN,ConvBlock=ResConvBlock,Conv=nn.Conv2d,is_chan_last=True` : uses a small
            ResNet.
            - `partial(UnetCNN,ConvBlock=ResConvBlock,Conv=nn.Conv2d,is_chan_last=True` : uses a
            UNet.

    kwargs :
        Additional arguments to `NeuralProcessFamily`.

    References
    ----------
    [1] Gordon, Jonathan, et al. "Convolutional conditional neural processes." arXiv preprint
    arXiv:1910.13556 (2019).
    """

    _valid_paths = ["deterministic"]

    def __init__(
        self,
        x_dim,
        y_dim,
        density_induced=128,
        Interpolator=SetConv,
        CNN=partial(
            CNN,
            ConvBlock=ResConvBlock,
            Conv=nn.Conv1d,
            n_blocks=3,
            Normalization=nn.Identity,
            is_chan_last=True,
            kernel_size=11,
        ),
        **kwargs,
    ):
        if (
            "Decoder" in kwargs and kwargs["Decoder"] != nn.Identity
        ):  # identity means that not using
            logger.warning(
                "`Decoder` was given to `ConvCNP`. To be translation equivariant you should disregard the first argument for example using `discard_ith_arg(Decoder, i=0)`, which is done by default when you DO NOT provide the Decoder."
            )

        # don't force det so that can inherit ,
        kwargs["encoded_path"] = kwargs.get("encoded_path", "deterministic")
        super().__init__(
            x_dim,
            y_dim,
            x_transf_dim=None,
            XEncoder=nn.Identity,
            **kwargs,
        )

        self.density_induced = density_induced
        # input is between -1 and 1 but use at least 0.5 temporary values on each sides to not
        # have strong boundary effects
        self.X_induced = torch.linspace(-1.5, 1.5, int(self.density_induced * 3))
        self.CNN = CNN

        self.cntxt_to_induced = Interpolator(self.x_dim, self.y_dim, self.r_dim)
        self.induced_to_induced = CNN(self.r_dim)
        self.induced_to_trgt = Interpolator(self.x_dim, self.r_dim, self.r_dim)

        self.reset_parameters()

    @property
    def n_induced(self):
        # using property because this might change after you set extrapolation
        return len(self.X_induced)

    @property
    def dflt_Modules(self):
        # allow inheritence
        dflt_Modules = ICNeuralProcessFamily.dflt_Modules.__get__(self)

        # don't depend on x
        dflt_Modules["Decoder"] = discard_ith_arg(dflt_Modules["SubDecoder"], i=0)

        return dflt_Modules

    def _get_X_induced(self, X):
        batch_size, _, _ = X.shape

        # effectively puts on cuda only once
        self.X_induced = self.X_induced.to(X.device)
        X_induced = self.X_induced.view(1, -1, 1)
        X_induced = X_induced.expand(batch_size, self.n_induced, self.x_dim)
        return X_induced

    def encode_globally(self, X_cntxt, Y_cntxt):
        batch_size, n_cntxt, _ = X_cntxt.shape

        # size = [batch_size, n_induced, x_dim]
        X_induced = self._get_X_induced(X_cntxt)

        # size = [batch_size, n_induced, r_dim]
        R_induced = self.cntxt_to_induced(X_cntxt, X_induced, Y_cntxt)

        if n_cntxt == 0:
            # arbitrarily setting the global representation to zero when no context
            # but the density channel will also be => makes sense
            R_induced = torch.zeros(
                batch_size, self.n_induced, self.r_dim, device=R_induced.device
            )

        # size = [batch_size, n_induced, r_dim]
        R_induced = self.induced_to_induced(R_induced)

        return R_induced

    def trgt_dependent_representation(self, X_cntxt, z_samples, R_induced, X_trgt):
        batch_size, n_trgt, _ = X_trgt.shape

        # size = [batch_size, n_induced, x_dim]
        X_induced = self._get_X_induced(X_cntxt)

        # size = [batch_size, n_trgt, r_dim]
        R_trgt = self.induced_to_trgt(X_induced, X_trgt, R_induced)

        # n_z_samples=1. size = [1, batch_size, n_trgt, r_dim]
        return R_trgt.unsqueeze(0)

    def set_extrapolation(self, min_max):
        """
        Scale the induced inputs to be in a given range while keeping
        the same density than during training (used for extrapolation.).
        """
        current_min = min_max[0] - 0.5
        current_max = min_max[1] + 0.5
        self.X_induced = torch.linspace(
            current_min,
            current_max,
            int(self.density_induced * (current_max - current_min)),
        )
