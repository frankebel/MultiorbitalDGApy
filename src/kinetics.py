import numpy as np

import brillouin_zone as bz
from i_have_mat import IHaveMat


class HoppingElement:
    def __init__(self, r_lat: list, orbs: list, value: float = 0.0):
        if not (isinstance(r_lat, list) and len(r_lat) == 3 and all(isinstance(x, int) for x in r_lat)):
            raise ValueError("'r_lat' must be a list with exactly 3 integer elements.")
        if not (
            isinstance(orbs, list)
            and len(orbs) == 2
            and all(isinstance(x, int) for x in orbs)
            and all(orb > 0 for orb in orbs)
        ):
            raise ValueError("'orbs' must be a list with exactly 2 integer elements that are greater than 0.")
        if not isinstance(value, (int, float)):
            raise ValueError("'value' must be a valid number.")

        self.r_lat = tuple(r_lat)
        self.orbs = np.array(orbs, dtype=int)
        self.value = float(value)


class Kinetics(IHaveMat):
    def __init__(self, mat: np.ndarray, er_r_grid: np.ndarray, er_r_weights: np.ndarray, er_orbs: np.ndarray):
        super().__init__(mat)
        self._er_r_grid = er_r_grid
        self._er_r_weights = er_r_weights
        self._er_orbs = er_orbs

    def get_ek(self, k_grid: bz.KGrid) -> np.ndarray:
        ek = self._convham_2_orbs(k_grid.kmesh.reshape(3, -1))
        n_orbs = ek.shape[-1]
        return ek.reshape(*k_grid.nk, n_orbs, n_orbs)

    def _convham_2_orbs(self, k_mesh: np.ndarray = None) -> np.ndarray:
        fft_grid = np.exp(1j * np.matmul(self._er_r_grid, k_mesh)) / self._er_r_weights[:, None, None]
        return np.transpose(np.sum(fft_grid * self.mat[..., None], axis=0), axes=(2, 0, 1))
