import numpy as np
import scipy

from matsubara_frequency_helper import MFHelper
from local_two_point import LocalTwoPoint
from local_self_energy import LocalSelfEnergy

import config


class LocalGreensFunction(LocalTwoPoint):
    def __init__(
        self,
        mat: np.ndarray,
        sigma: LocalSelfEnergy = None,
        ek: np.ndarray = None,
        full_niv_range: bool = True,
    ):
        super().__init__(mat, full_niv_range)
        self._sigma = sigma
        self._ek = ek

        config.n, config.n_of_k = (0, 0) if ek is None else self.get_fill()

    @staticmethod
    def create_from_dmft(mat: np.ndarray) -> "LocalGreensFunction":
        mat = np.einsum("i...,ij->ij...", mat, np.eye(mat.shape[0]))
        return LocalGreensFunction(mat)

    def get_fill(self) -> (float, np.ndarray):
        self.mat = self.get_gloc_mat()
        g_model = self.get_g_model_mat()
        hloc: np.ndarray = np.mean(self._ek, axis=(0, 1, 2))
        smom0, _ = self._sigma.smom

        mu_bands: np.ndarray = config.mu * np.eye(self.n_bands)
        if (config.beta * np.linalg.eigvals(hloc.real + smom0 - mu_bands) < 20).any():
            rho_loc = np.linalg.inv(
                np.eye(self.n_bands) + scipy.linalg.expm(config.beta * (hloc.real + smom0 - mu_bands))
            )
        else:
            rho_loc = scipy.linalg.expm(-config.beta * (smom0 + hloc.real - mu_bands))

        filling_k = rho_loc + np.sum(self.mat.real - g_model.real, axis=-1) / config.beta
        n_el = 2.0 * np.trace(filling_k).real
        return n_el, filling_k

    def get_gloc_mat(self):
        eye_bands = np.eye(self.n_bands, self.n_bands)
        v = MFHelper.get_ivn(self.niv, config.beta)
        iv_bands = (v[None, None, :] * eye_bands[..., None])[None, None, None, ...]
        mu_bands = (config.mu * eye_bands)[None, None, None, ..., None]
        mat = iv_bands + mu_bands - self._ek[..., None] - self._sigma.mat[None, None, None, :]
        mat = np.linalg.inv(mat.transpose(0, 1, 2, 5, 3, 4)).transpose(0, 1, 2, 4, 5, 3)

        return np.mean(mat, axis=(0, 1, 2))

    def get_g_model_mat(self):
        eye_bands = np.eye(self.n_bands, self.n_bands)
        v = MFHelper.get_ivn(self.niv, config.beta)
        iv_bands = v[None, None, :] * eye_bands[..., None]
        mu_bands = (config.mu * eye_bands)[..., None]
        hloc: np.ndarray = np.mean(self._ek, axis=(0, 1, 2))

        smom0, _ = self._sigma.smom

        mat = iv_bands + mu_bands - hloc - smom0
        return np.linalg.inv(mat.transpose(2, 0, 1)).transpose(1, 2, 0)
