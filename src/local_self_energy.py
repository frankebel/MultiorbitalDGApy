import numpy as np
from local_two_point import LocalTwoPoint
from matsubara_frequency_helper import MFHelper

import config


class LocalSelfEnergy(LocalTwoPoint):
    def __init__(self, mat: np.ndarray, full_niv_range: bool = True, do_smom_fit: bool = False):
        super().__init__(mat, full_niv_range=full_niv_range)
        if do_smom_fit:
            self._smom0, self._smom1 = self._fit_smom()

    @property
    def smom(self) -> (float, float):
        return self._smom0, self._smom1

    @staticmethod
    def from_dmft(mat: np.ndarray) -> "LocalSelfEnergy":
        mat = np.einsum("i...,ij->ij...", mat, np.eye(mat.shape[0]))
        return LocalSelfEnergy(mat, do_smom_fit=True)

    def _fit_smom(self):
        mat_half_v = self.mat[..., self.niv :]
        iv = MFHelper.get_ivn(self.niv, config.beta, return_only_positive=True)

        n_freq_fit = int(0.2 * self.niv)
        if n_freq_fit < 4:
            n_freq_fit = 4

        iwfit = iv[self.niv - n_freq_fit :][None, None, :] * np.eye(self.n_bands)[:, :, None]
        fitdata = mat_half_v[..., self.niv - n_freq_fit :]

        mom0 = np.mean(fitdata.real, axis=-1)
        mom1 = np.mean(fitdata.imag * iwfit.imag, axis=-1)
        return mom0, mom1
