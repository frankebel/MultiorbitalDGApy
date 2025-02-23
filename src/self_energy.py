import numpy as np
from local_two_point import LocalTwoPoint
from matsubara_frequencies import MFHelper

import config
import itertools as it


class SelfEnergy(LocalTwoPoint):
    def __init__(self, mat: np.ndarray, full_niv_range: bool = True):
        super().__init__(mat, full_niv_range=full_niv_range)
        # TODO: check if this is a reasonable value. I'd suggest it depends on the input data size.
        self.niv_core_min = 20
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

    def _estimate_niv_core(self, err: float = 1e-4):
        """Check when the real and the imaginary part are within error margin of the asymptotic"""
        asympt = self._get_asympt(niv_asympt=self.niv, n_min=0)

        max_ind_real = 0
        max_ind_imag = 0

        for i, j in it.product(range(self.n_bands), repeat=2):
            k_mean = np.mean(self.mat[:, :, :, i, j, :], axis=(0, 1, 2))
            ind_real = np.argmax(np.abs(k_mean.real - asympt.mat.real) < err)
            ind_imag = np.argmax(np.abs(k_mean.imag - asympt.mat.imag) < err)

            max_ind_real = max(max_ind_real, ind_real)
            max_ind_imag = max(max_ind_imag, ind_imag)

        niv_core = max(max_ind_real, max_ind_imag)
        if niv_core < self.niv_core_min:
            return self.niv_core_min
        return niv_core

    def _get_asympt(self, niv_asympt: int, n_min: int = None):
        """
        Returns purely the asymptotic behaviour of the self-energy for the given frequency range niv_asympt.
        Not intended to be used as its own but intended to be padded to the self-energy as an asymptotic tail.
        """
        if n_min is None:
            n_min = self.niv
        iv_asympt = 1j * MFHelper.vn(niv_asympt, config.sys.beta, shift=n_min, return_only_positive=True)
        asympt = (self._smom0 - 1.0 / iv_asympt * self._smom1)[None, None, None, ...] * np.ones(config.lattice.nk)[
            ..., None, None, None
        ]
        return SelfEnergy(asympt, full_niv_range=False)

    def extend_to_multi_orbital(self, padding_object: "SelfEnergy", n_bands: int) -> "SelfEnergy":
        """
        Mainly for testing. Extends the single-orbital object to a multi-orbital object, putting padding_mat in the new bands.
        """
        if (
            self.num_orbital_dimensions != padding_object.num_orbital_dimensions
            or self.num_wn_dimensions != padding_object.num_wn_dimensions
            or self.num_vn_dimensions != padding_object.num_vn_dimensions
        ):
            raise ValueError("Number of orbital, bosonic or fermionic frequency dimensions do not match.")

        shape = (n_bands,) * self.num_orbital_dimensions + (1,) * self.num_wn_dimensions + (1,) * self.num_vn_dimensions

        new_mat = np.tile(padding_object.mat, shape)
        new_mat[0, 0, ...] = self.mat[0, 0, ...]
        return SelfEnergy(new_mat, True)

    def __add__(self, other):
        if not isinstance(other, SelfEnergy):
            raise ValueError("Can only add two SelfEnergy objects.")
        if self.original_shape != other.original_shape:
            raise ValueError("Cannot add two SelfEnergy objects of different shapes.")
        return SelfEnergy(self.mat + other.mat, full_niv_range=self.full_niv_range)

    def __sub__(self, other):
        return self.__add__(-other)
