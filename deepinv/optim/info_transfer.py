import math
import pywt
import torch
# import ptwt as pywt


class InfoTransfer:
    def __init__(self, x=None):
        self.cit_c = None
        self.cit_r = None
        if isinstance(x, torch.Tensor):
            self.build_cit_matrices(x)

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


class WaveletTransfer(InfoTransfer):
    def __init__(self, wavelet_type, x=None):
        super().__init__()
        self.wavelet_type = wavelet_type

    def projection(self, x):
        cit_c, cit_r = self.get_cit_matrices()

        n_chan = x.shape[1]
        n_row_coa = cit_c.shape[2]
        n_col_coa = cit_r.shape[2]

        x_coa = torch.zeros([1, n_chan, n_row_coa, n_col_coa], device=x.device).type(
            x.dtype
        )
        for chan in range(0, n_chan):
            x_coa[0, chan, :, :] = cit_c @ x[0, chan, :, :] @ cit_r.transpose(2, 3)

        return x_coa

    def prolongation(self, x):
        cit_c, cit_r = self.get_cit_matrices()

        n_chan = x.shape[1]
        n_row_fine = cit_c.shape[3]
        n_col_fine = cit_r.shape[3]

        x_fine = torch.zeros([1, n_chan, n_row_fine, n_col_fine], device=x.device).type(
            x.dtype
        )
        for chan in range(0, n_chan):
            x_fine[0, chan, :, :] = cit_c.transpose(2, 3) @ x[0, chan, :, :] @ cit_r

        return x_fine

    @staticmethod
    def _cit_convolution_matrix(mat, qmf):
        m_row = mat.shape[2]
        m_col = mat.shape[3]

        len_qmf = len(qmf)  # number of coefficients

        # Build first row
        if len_qmf >= m_col:
            print(f"[Warning] len_qmf = {len_qmf} >= {m_col} = n_row")

        n0 = min(len_qmf, m_col)
        mat[0, 0, 0, 0:n0] = torch.tensor(qmf[0:n0])

        # Finish building the QMF convolution matrix as a decimated Toeplitz matrix
        ic = 0
        for ir in range(1, m_row):
            ic += 2
            if ic >= m_col:
                break

            ic0 = m_col - ic
            mat[0, 0, ir, ic:] = mat[0, 0, 0, 0:ic0]

    def build_cit_matrices(self, img, ratio=2):
        n_row = img.shape[2]
        n_col = img.shape[3]

        w_vec = pywt.Wavelet(self.wavelet_type).dec_lo
        w_norm = math.sqrt(sum([x**2 for x in w_vec]))
        w_dec_lo = [x / w_norm for x in w_vec]

        n_row_coa = int(torch.floor(torch.tensor(img.shape[2] / ratio)))
        n_col_coa = int(torch.floor(torch.tensor(img.shape[3] / ratio)))

        # Define QMF convolution matrix
        conv_qmf_col = torch.zeros([1, 1, n_row_coa, n_row], device=img.device).type(
            img.dtype
        )
        self._cit_convolution_matrix(conv_qmf_col, w_dec_lo)
        conv_qmf_row = torch.zeros([1, 1, n_col_coa, n_col], device=img.device).type(
            img.dtype
        )
        self._cit_convolution_matrix(conv_qmf_row, w_dec_lo)

        self.cit_c = conv_qmf_col
        self.cit_r = conv_qmf_row

        return conv_qmf_col, conv_qmf_row
