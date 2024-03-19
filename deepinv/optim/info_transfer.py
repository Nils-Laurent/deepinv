import math
import pywt
import torch
from deepinv.physics import Downsampling


class InfoTransfer:
    def __init__(self, x):
        pass

    def build_cit_matrices(self, x):
        raise NotImplementedError("cit_matrices not overridden")

    def get_cit_matrices(self):
        if self.cit_c is None or self.cit_r is None:
            raise ValueError("CIT matrix is none")
        return self.cit_c, self.cit_r

    def projection(self, x):
        raise NotImplementedError("projection not overridden")

    def prolongation(self, x):
        raise NotImplementedError("prolongation not overridden")


class DownsamplingTransfer(InfoTransfer):
    def __init__(self, x, def_filter='kaiser'):
        super().__init__(x)

        if def_filter == 'kaiser':
            k0 = self.get_kaiser()
        elif def_filter == 'blackmanharris':
            k0 = self.get_blackmanharris()
        else:
            raise NotImplementedError("Downsampling filter not implemented")
        filt = self.set_filter(k0)
        self.op = Downsampling(x.shape, filter=filt, factor=2, device=x.device, padding="circular")

    def set_filter(self, k0):
        k0 = k0 / torch.sum(k0)
        k_filter = torch.outer(k0, k0)
        k_filter = k_filter.unsqueeze(0).unsqueeze(0)
        return k_filter

    def get_kaiser(self):
        # N = 10
        # beta = 10.0
        k0 = torch.tensor([
            0.0004, 0.0310, 0.2039, 0.5818, 0.9430,
            0.9430, 0.5818, 0.2039, 0.0310, 0.0004
        ])
        return k0

    def get_blackmanharris(self):
        # 8 coefficients
        k0 = torch.tensor(
            [0.0001, 0.0334, 0.3328, 0.8894,
             0.8894, 0.3328, 0.0334, 0.0001]
        )
        return k0

    def projection(self, x):
        return self.op.A(x)

    def prolongation(self, x):
        return self.op.A_adjoint(x)
