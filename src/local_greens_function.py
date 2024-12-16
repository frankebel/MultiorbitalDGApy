import numpy as np
import scipy

import config
from local_self_energy import LocalSelfEnergy
from local_two_point import LocalTwoPoint
from matsubara_frequencies import MFHelper


class LocalGreensFunction(LocalTwoPoint):
    """
    Represents a momentum-independent Green's function.
    """

    def __init__(
        self,
        mat: np.ndarray,
        sigma: LocalSelfEnergy = None,
        ek: np.ndarray = None,
        full_niv_range: bool = True,
    ):
        super().__init__(mat, full_niv_range=full_niv_range)
        self._sigma = sigma
        self._ek = ek

        if sigma is not None and ek is not None:
            config.n, config.rho_orbs = self._get_fill()

    @staticmethod
    def from_dmft(mat: np.ndarray) -> "LocalGreensFunction":
        """
        Creates a LocalGreensFunction object from a given DMFT file input matrix.

        """
        mat = np.einsum("i...,ij->ij...", mat, np.eye(mat.shape[0]), optimize=True)
        return LocalGreensFunction(mat)

    @staticmethod
    def create_g_loc(siw: LocalSelfEnergy, ek: np.ndarray) -> "LocalGreensFunction":
        """
        Returns a local Green's function object from a given self-energy and band dispersion.
        """
        return LocalGreensFunction(np.empty_like(siw.mat), siw, ek, siw.full_niv_range)

    @property
    def e_kin(self):  # be careful about the on-site energy
        """
        Returns the kinetic energy of the system.
        """
        ekin = 1 / config.beta * np.sum(np.mean(self._ek[..., None] * self.mat, axis=(0, 1, 2)))
        assert np.abs(ekin.imag) < 1e-8, "Kinetic energy must be real."
        return ekin.real

    def _get_fill(self) -> (float, np.ndarray):
        """
        Returns the total filling and the filling of each band.
        """
        self.mat = self._get_gloc_mat()
        g_model = self._get_g_model_mat()
        hloc: np.ndarray = np.mean(self._ek, axis=(0, 1, 2))
        smom0, _ = self._sigma.smom
        mu_bands: np.ndarray = config.mu * np.eye(self.n_bands)

        test = config.beta * (hloc.real + smom0 - mu_bands)

        eigenvals, eigenvecs = np.linalg.eig(config.beta * (hloc.real + smom0 - mu_bands))
        rho_loc_diag = np.zeros((self.n_bands, self.n_bands), dtype=np.complex64)
        for i in range(self.n_bands):
            if eigenvals[i] > 0:
                rho_loc_diag[i, i] = np.exp(-eigenvals[i]) / (1 + np.exp(-eigenvals[i]))
            else:
                rho_loc_diag[i, i] = 1 / (1 + np.exp(eigenvals[i]))

        rho_loc = eigenvecs @ rho_loc_diag @ np.linalg.inv(eigenvecs)
        rho_new = rho_loc + np.sum(self.mat.real - g_model.real, axis=-1) / config.beta
        n_el = 2.0 * np.trace(rho_new).real
        return n_el, rho_new

    def _get_gloc_mat(self):
        iv_bands, mu_bands = self._get_g_params_local()
        iv_bands = iv_bands[None, None, None, ...]
        mu_bands = mu_bands[None, None, None, ...]

        mat = iv_bands + mu_bands - self._ek[..., None] - self._sigma.mat[None, None, None, :]
        mat = np.linalg.inv(mat.transpose(0, 1, 2, 5, 3, 4)).transpose(0, 1, 2, 4, 5, 3)
        return np.mean(mat, axis=(0, 1, 2))

    def _get_g_model_mat(self):
        iv_bands, mu_bands = self._get_g_params_local()
        hloc: np.ndarray = np.mean(self._ek, axis=(0, 1, 2))
        smom0, _ = self._sigma.smom
        mat = iv_bands + mu_bands - hloc[..., None] - smom0[..., np.newaxis]
        return np.linalg.inv(mat.transpose(2, 0, 1)).transpose(1, 2, 0)

    def _get_g_params_local(self):
        eye_bands = np.eye(self.n_bands, self.n_bands)
        iv = 1j * MFHelper.vn(self.niv, config.beta)
        iv_bands = iv[None, None, :] * eye_bands[..., None]
        mu_bands = config.mu * eye_bands[:, :, None]
        return iv_bands, mu_bands
