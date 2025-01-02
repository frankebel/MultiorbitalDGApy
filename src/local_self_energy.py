import numpy as np
from local_two_point import LocalTwoPoint
from matsubara_frequencies import MFHelper

import config


class LocalSelfEnergy(LocalTwoPoint):
    def __init__(self, mat: np.ndarray, full_niv_range: bool = True):
        super().__init__(mat, full_niv_range=full_niv_range)
        self._smom0, self._smom1 = self._fit_smom()

    @property
    def smom(self) -> (float, float):
        return self._smom0, self._smom1

    def _fit_smom(self):
        mat_half_v = self.mat[..., self.niv :]
        iv = 1j * MFHelper.vn(self.niv, config.sys.beta, return_only_positive=True)

        n_freq_fit = int(0.2 * self.niv)
        if n_freq_fit < 4:
            n_freq_fit = 4

        iwfit = iv[self.niv - n_freq_fit :][None, None, :] * np.eye(self.n_bands)[:, :, None]
        fitdata = mat_half_v[..., self.niv - n_freq_fit :]

        mom0 = np.mean(fitdata.real, axis=-1)
        mom1 = np.mean(fitdata.imag * iwfit.imag, axis=-1)
        return mom0, mom1

    def extend_to_multi_orbital(self, padding_object: "LocalSelfEnergy", n_bands: int) -> "LocalSelfEnergy":
        """
        Mainly for testing. Extends the single-orbital object to a multi-orbital object, putting padding_mat in the new bands.
        """
        if (
            self.num_orbital_dimensions != padding_object.num_orbital_dimensions
            or self.num_bosonic_frequency_dimensions != padding_object.num_bosonic_frequency_dimensions
            or self.num_fermionic_frequency_dimensions != padding_object.num_fermionic_frequency_dimensions
        ):
            raise ValueError("Number of orbital, bosonic or fermionic frequency dimensions do not match.")

        shape = (
            (n_bands,) * self.num_orbital_dimensions
            + (1,) * self.num_bosonic_frequency_dimensions
            + (1,) * self.num_fermionic_frequency_dimensions
        )

        new_mat = np.tile(padding_object.mat, shape)
        new_mat[0, 0, ...] = self.mat[0, 0, :]
        return LocalSelfEnergy(new_mat, True)
