import itertools as it
from copy import deepcopy

import numpy as np

import config
from matsubara_frequencies import MFHelper
from n_point_base import IAmNonLocal
from local_n_point import LocalNPoint


class SelfEnergy(LocalNPoint, IAmNonLocal):
    """
    Represents the self-energy. Will automatically map to full niv range if full_niv_range is set to False.
    """

    def __init__(
        self,
        mat: np.ndarray,
        nk: tuple[int, int, int] = (1, 1, 1),
        full_niv_range: bool = True,
        has_compressed_momentum_dimension: bool = False,
        estimate_niv_core: bool = False,
    ):
        LocalNPoint.__init__(self, mat, 2, 0, 1, full_niv_range=full_niv_range)
        IAmNonLocal.__init__(self, mat, nk, has_compressed_momentum_dimension=has_compressed_momentum_dimension)
        # TODO: check if this is a reasonable value. I'd suggest it depends on the input data size.
        self._niv_core_min = 20

        if not full_niv_range:
            self.to_full_niv_range()

        self._smom0, self._smom1 = self._fit_smom()
        self._niv_core = self._estimate_niv_core() if estimate_niv_core else self.niv

    @property
    def smom(self) -> tuple[float, float]:
        """
        Returns the first two local momenta of the self-energy.
        """
        return self._smom0, self._smom1

    @property
    def n_bands(self) -> int:
        """
        Returns the number of bands.
        """
        return self.original_shape[1] if self.has_compressed_q_dimension else self.original_shape[3]

    def _fit_smom(self):
        """
        Fits the first two local momenta of the self-energy.
        """
        mat_half_v = np.mean(self.mat[..., self.niv :], axis=(0, 1, 2))
        iv = 1j * MFHelper.vn(self.niv, config.sys.beta, return_only_positive=True)

        n_freq_fit = int(0.2 * self.niv)
        if n_freq_fit < 4:
            n_freq_fit = 4

        iwfit = iv[self.niv - n_freq_fit :][None, None, :] * np.eye(self.n_bands)[:, :, None]
        fitdata = mat_half_v[..., self.niv - n_freq_fit :]

        mom0 = np.mean(fitdata.real, axis=-1)
        mom1 = np.mean(fitdata.imag * iwfit.imag, axis=-1)
        return mom0, mom1

    def _estimate_niv_core(self, err: float = 1e-5):
        """
        Check when the real and the imaginary part are within an error margin of the asymptotic.
        """
        asympt = self._get_asympt(niv_full=self.niv, n_min=0)

        max_ind_real = 0
        max_ind_imag = 0

        for i, j in it.product(range(self.n_bands), repeat=2):
            k_mean = np.mean(self.mat[:, :, :, i, j, :], axis=(0, 1, 2))
            asympt_mean = np.mean(asympt.mat[:, :, :, i, j, :], axis=(0, 1, 2))
            ind_real = np.argmax(np.abs(k_mean.real - asympt_mean.real) < err)
            ind_imag = np.argmax(np.abs(k_mean.imag - asympt_mean.imag) < err)

            max_ind_real = max(max_ind_real, ind_real)
            max_ind_imag = max(max_ind_imag, ind_imag)

        niv_core = max(max_ind_real, max_ind_imag)
        if niv_core < self._niv_core_min:
            return self._niv_core_min
        return niv_core

    def _get_asympt(self, niv_full: int, n_min: int = None) -> "SelfEnergy":
        """
        Returns purely the asymptotic behaviour of the self-energy for the given frequency range.
        Not intended to be used as its own but intended to be padded to the self-energy as an asymptotic tail.
        """
        if n_min is None:
            n_min = self.niv
        iv_asympt = 1j * MFHelper.vn(niv_full, config.sys.beta, shift=n_min)[None, None, ...]
        asympt = (self._smom0[..., None] - 1.0 / iv_asympt * self._smom1[..., None])[None, None, None, ...] * np.ones(
            self.nq
        )[..., None, None, None]
        return SelfEnergy(asympt)

    def create_with_asympt(self) -> "SelfEnergy":
        """
        Concatenates the core and the asymptotic tail of the self-energy.
        """
        copy = deepcopy(self)
        asympt = copy._get_asympt(niv_full=copy.niv)

        if copy._niv_core == copy.niv:
            return copy
        if asympt.niv == 0:
            return copy

        copy = copy.cut_niv(copy._niv_core)
        copy.mat = np.concatenate(
            (asympt.mat[..., : asympt.niv - copy.niv], copy.mat, asympt.mat[..., asympt.niv + copy.niv :]), axis=-1
        )
        return copy

    def extend_to_multi_orbital(self, padding_object: "SelfEnergy", n_bands: int) -> "SelfEnergy":
        """
        Mainly for testing. Extends the single-orbital object to a multi-orbital object, putting padding_object in the new bands.
        """
        if (
            self.num_orbital_dimensions != padding_object.num_orbital_dimensions
            or self.num_wn_dimensions != padding_object.num_wn_dimensions
            or self.num_vn_dimensions != padding_object.num_vn_dimensions
        ):
            raise ValueError("Number of orbital, bosonic or fermionic frequency dimensions do not match.")

        shape = (n_bands,) * self.num_orbital_dimensions + (1,) * (self.num_wn_dimensions + self.num_vn_dimensions)

        new_mat = np.tile(padding_object.mat, shape)
        new_mat[0, 0, ...] = self.mat[0, 0, ...]
        return SelfEnergy(new_mat[None, None, None, ...])

    def __add__(self, other):
        """
        Adds two SelfEnergy objects.
        """
        return self.add(other)

    def __sub__(self, other):
        """
        Subtracts two SelfEnergy objects.
        """
        return self.sub(other)

    def add(self, other) -> "SelfEnergy":
        """
        Adds two SelfEnergy objects.
        """
        if not isinstance(other, (SelfEnergy, np.ndarray)):
            raise ValueError(f"Can not add {type(other)} to {type(self)}.")

        if isinstance(other, np.ndarray):
            return SelfEnergy(
                self.mat + other,
                full_niv_range=self.full_niv_range,
                has_compressed_momentum_dimension=self.has_compressed_q_dimension,
            )

        self._align_q_dimensions_for_operations(other)
        return SelfEnergy(
            self.mat + other.mat, full_niv_range=self.full_niv_range, has_compressed_momentum_dimension=True
        )

    def sub(self, other) -> "SelfEnergy":
        """
        Subtracts two SelfEnergy objects.
        """
        return self.add(-other)

    def pad_with_dmft_self_energy(self, other: "SelfEnergy") -> "SelfEnergy":
        """
        Pads the self-energy with the other self-energy up to niv.
        """
        if self.niv > other.niv:
            raise ValueError("Can not pad with a self-energy that has less frequencies.")
        niv_diff = other.niv - self.niv

        self.compress_q_dimension()
        other.compress_q_dimension()

        other_mat = np.tile(other.mat, (self.nq_tot, 1, 1, 1))
        result_mat = np.concatenate(
            (other_mat[..., :niv_diff], self.mat, other_mat[..., niv_diff + 2 * self.niv :]), axis=-1
        )
        return SelfEnergy(result_mat, self.nq, self.full_niv_range, self.has_compressed_q_dimension, False)
