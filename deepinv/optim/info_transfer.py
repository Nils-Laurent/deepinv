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
            filt = self.get_kaiser()
        else:
            raise NotImplementedError("Downsampling filter not implemented")

        self.op = Downsampling(x.shape, filter=filt, factor=2, device=x.device, padding="circular")

    def get_kaiser(self):
        # beta = 1
        k0 = torch.tensor([0.7898,0.9214,0.9911,0.9911,0.9214,0.7898])
        k0 = k0 / torch.sum(k0)
        k_filter = torch.outer(k0, k0)
        k_filter = k_filter.unsqueeze(0).unsqueeze(0)
        return k_filter

    def projection(self, x):
        return self.op.A(x)

    def prolongation(self, x):
        return self.op.A_adjoint(x)
