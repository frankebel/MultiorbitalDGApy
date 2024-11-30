import os
from typing import Any

import numpy as np
from numpy import ndarray, dtype

import brillouin_zone as bz
import itertools as it
import pandas as pd


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


class InteractionElement:
    def __init__(self, r_lat: list, orbs: list, value: float = 0.0):
        if not isinstance(r_lat, list) and len(r_lat) == 3 and all(isinstance(x, int) for x in r_lat):
            raise ValueError("'r_lat' must be a list with exactly 3 integer elements.")
        if (
            not isinstance(orbs, list)
            and len(orbs) == 4
            and all(isinstance(x, int) for x in orbs)
            and all(orb > 0 for orb in orbs)
        ):
            raise ValueError("'orbs' must be a list with exactly 4 integer elements that are greater than zero.")
        if not isinstance(value, (int, float)):
            raise ValueError("'value' must be a valid number.")

        self.r_lat = tuple(r_lat)
        self.orbs = np.array(orbs, dtype=int)
        self.value = float(value)


class Hamiltonian:
    def __init__(self):
        self._er = None
        self._er_r_grid = None
        self._er_r_weights = None
        self._er_orbs = None
        self._local_ur = None
        self._nonlocal_ur = None
        self._ur_r_grid = None
        self._ur_r_weights = None
        self._ur_orbs = None

    @property
    def er(self) -> np.ndarray:
        return self._er

    @property
    def local_ur(self) -> np.ndarray:
        return self._local_ur

    @property
    def nonlocal_ur(self) -> np.ndarray:
        return self._nonlocal_ur

    @property
    def er_r_grid(self) -> np.ndarray:
        return self._er_r_grid

    @property
    def er_r_weights(self) -> np.ndarray:
        return self._er_r_weights

    @property
    def er_orbs(self) -> np.ndarray:
        return self._er_orbs

    @property
    def ur_r_grid(self) -> np.ndarray:
        return self._ur_r_grid

    @property
    def ur_r_weights(self) -> np.ndarray:
        return self._ur_r_weights

    @property
    def ur_orbs(self) -> np.ndarray:
        return self._ur_orbs

    def set_kinetic_properties(
        self, er: np.ndarray, er_r_grid: np.ndarray, er_r_weights: np.ndarray, er_orbs: np.ndarray
    ) -> None:
        self._er = er
        self._er_r_grid = er_r_grid
        self._er_r_weights = er_r_weights
        self._er_orbs = er_orbs

    def set_interaction_properties(
        self,
        local_ur: np.ndarray,
        nonlocal_ur: np.ndarray,
        ur_r_grid: np.ndarray,
        ur_r_weights: np.ndarray,
        ur_orbs: np.ndarray,
    ) -> None:
        self._local_ur = local_ur
        self._nonlocal_ur = nonlocal_ur
        self._ur_r_grid = ur_r_grid
        self._ur_r_weights = ur_r_weights
        self._ur_orbs = ur_orbs

    def get_ek(self, k_grid: bz.KGrid) -> np.ndarray:
        ek = self.convham_2_orbs(self._er, self._er_r_grid, self._er_r_weights, k_grid.kmesh.reshape(3, -1))
        n_orbs = ek.shape[-1]
        return ek.reshape(*k_grid.nk, n_orbs, n_orbs)

    def get_local_uk(self) -> np.ndarray:
        return self.local_ur

    def get_nonlocal_uk(self, k_grid: bz.KGrid) -> np.ndarray:
        uk = self.convham_4_orbs(self._nonlocal_ur, self._ur_r_grid, self._ur_r_weights, k_grid.kmesh.reshape(3, -1))
        n_orbs = uk.shape[-1]
        return uk.reshape(*k_grid.nk, n_orbs, n_orbs, n_orbs, n_orbs)

    def convham_2_orbs(
        self,
        er: np.ndarray = None,
        er_r_grid: np.ndarray = None,
        er_r_weights: np.ndarray = None,
        k_mesh: np.ndarray = None,
    ) -> np.ndarray:
        fft_grid = np.exp(1j * np.matmul(er_r_grid, k_mesh)) / er_r_weights[:, None, None]
        return np.transpose(np.sum(fft_grid * er[..., None], axis=0), axes=(2, 0, 1))

    def convham_4_orbs(
        self,
        nonlocal_ur: np.ndarray = None,
        ur_r_grid: np.ndarray = None,
        ur_r_weights: np.ndarray = None,
        k_mesh: np.ndarray = None,
    ) -> np.ndarray:
        fft_grid = np.exp(1j * np.matmul(ur_r_grid, k_mesh)) / ur_r_weights[:, None, None, None, None]
        return np.transpose(np.sum(fft_grid * nonlocal_ur[..., None], axis=0), axes=(4, 0, 1, 2, 3))


class HamiltonianBuilder:
    def __init__(self):
        self._real_space_hamiltonian = Hamiltonian()

    def build(self) -> Hamiltonian:
        return self._real_space_hamiltonian

    def add(self, hopping_elements: list, interaction_elements: list) -> "HamiltonianBuilder":
        return self.add_kinetic(hopping_elements).add_interaction(interaction_elements)

    def add_kinetic(self, hopping_elements: list) -> "HamiltonianBuilder":
        if not all(isinstance(item, HoppingElement) for item in hopping_elements):
            hopping_elements = self.parse_to_hopping_elements(hopping_elements)

        if any(np.allclose(he.r_lat, [0, 0, 0]) for he in hopping_elements):
            raise ValueError("Local hopping is not allowed!")

        unique_lat_r = set([he.r_lat for he in hopping_elements])
        r2ind = {tup: index for index, tup in enumerate(unique_lat_r)}

        n_rp = len(r2ind)
        n_orbs = int(max(np.array([he.orbs for he in hopping_elements]).flatten()))
        er_r_grid = self.create_er_grid(r2ind, n_orbs)

        er_orbs = self.create_er_orbs(n_rp, n_orbs)
        er_r_weights = np.ones(n_rp)[:, None]

        er = np.zeros((n_rp, n_orbs, n_orbs))
        for he in hopping_elements:
            self.insert_er_element(er, r2ind, he.r_lat, *he.orbs, he.value)

        self._real_space_hamiltonian.set_kinetic_properties(er, er_r_grid, er_r_weights, er_orbs)
        return self

    def add_interaction(self, interaction_elements: list) -> "HamiltonianBuilder":
        if not all(isinstance(item, InteractionElement) for item in interaction_elements):
            interaction_elements = self.parse_to_interaction_elements(interaction_elements)

        unique_r_lat = set([he.r_lat for he in interaction_elements])
        r2ind = {tup: index for index, tup in enumerate(unique_r_lat)}

        n_rp = len(r2ind)
        n_orbs = int(max(np.array([he.orbs for he in interaction_elements]).flatten()))
        ur_nonlocal_r_grid = self.create_ur_grid(r2ind, n_orbs)

        ur_orbs = self.create_ur_orbs(n_rp, n_orbs)
        ur_nonlocal_r_weights = np.ones(n_rp)[:, None]

        ur_local = np.zeros((n_orbs, n_orbs, n_orbs, n_orbs))
        ur_nonlocal = np.zeros((n_rp, n_orbs, n_orbs, n_orbs, n_orbs))
        for ie in interaction_elements:
            if np.allclose(ie.r_lat, [0, 0, 0]):
                self.insert_ur_element(ur_local, None, None, *ie.orbs, ie.value)
            else:
                self.insert_ur_element(ur_nonlocal, r2ind, ie.r_lat, *ie.orbs, ie.value)

        self._real_space_hamiltonian.set_interaction_properties(
            ur_local, ur_nonlocal, ur_nonlocal_r_grid, ur_nonlocal_r_weights, ur_orbs
        )
        return self

    def add_single_band_interaction(self, u: float) -> "HamiltonianBuilder":
        interaction_elements = [InteractionElement([0, 0, 0], [1, 1, 1, 1], u)]
        return self.add_interaction(interaction_elements)

    def add_kinetic_one_band_2d_t_tp_tpp(self, t: float, tp: float, tpp: float) -> "HamiltonianBuilder":
        orbs = [1, 1]
        hopping_elements = [
            HoppingElement(r_lat=[1, 0, 0], orbs=orbs, value=-t),
            HoppingElement(r_lat=[0, 1, 0], orbs=orbs, value=-t),
            HoppingElement(r_lat=[-1, 0, 0], orbs=orbs, value=-t),
            HoppingElement(r_lat=[0, -1, 0], orbs=orbs, value=-t),
            HoppingElement(r_lat=[1, 1, 0], orbs=orbs, value=-tp),
            HoppingElement(r_lat=[1, -1, 0], orbs=orbs, value=-tp),
            HoppingElement(r_lat=[-1, 1, 0], orbs=orbs, value=-tp),
            HoppingElement(r_lat=[-1, -1, 0], orbs=orbs, value=-tp),
            HoppingElement(r_lat=[2, 0, 0], orbs=orbs, value=-tpp),
            HoppingElement(r_lat=[0, 2, 0], orbs=orbs, value=-tpp),
            HoppingElement(r_lat=[-2, 0, 0], orbs=orbs, value=-tpp),
            HoppingElement(r_lat=[0, -2, 0], orbs=orbs, value=-tpp),
        ]

        return self.add_kinetic(hopping_elements)

    def add_kinetic_from_file(self, filepath: str = "./", filename: str = "wannier90.dat") -> "HamiltonianBuilder":
        hr_file = pd.read_csv(
            os.path.join(filepath, filename), skiprows=1, names=np.arange(15), sep=r"\s+", dtype=float, engine="python"
        )
        n_bands = hr_file.values[0][0].astype(int)
        nr = hr_file.values[1][0].astype(int)

        tmp = np.reshape(hr_file.values, (np.size(hr_file.values), 1))
        tmp = tmp[~np.isnan(tmp)]

        er_r_weights = tmp[2 : 2 + nr].astype(int)
        er_r_weights = np.reshape(er_r_weights, (np.size(er_r_weights), 1))
        ns = 7
        n_tmp = np.size(tmp[2 + nr :]) // ns
        tmp = np.reshape(tmp[2 + nr :], (n_tmp, ns))

        er_r_grid = np.reshape(tmp[:, 0:3], (nr, n_bands, n_bands, 3))
        er_orbs = np.reshape(tmp[:, 3:5], (nr, n_bands, n_bands, 2))
        er = np.reshape(tmp[:, 5] + 1j * tmp[:, 6], (nr, n_bands, n_bands))

        self._real_space_hamiltonian.set_kinetic_properties(er, er_r_grid, er_r_weights, er_orbs)
        return self

    def create_er_grid(self, r2ind: dict[list, int], n_orbs: int) -> np.ndarray:
        n_rp = len(r2ind)
        r_grid = np.zeros((n_rp, n_orbs, n_orbs, 3))
        for r_vec in r2ind.keys():
            r_grid[r2ind[r_vec], :, :, :] = r_vec
        return r_grid

    def create_ur_grid(self, r2ind: dict[list, int], n_orbs: int) -> np.ndarray:
        n_rp = len(r2ind)
        r_grid = np.zeros((n_rp, n_orbs, n_orbs, n_orbs, n_orbs, 3))
        for r_vec in r2ind.keys():
            r_grid[r2ind[r_vec], :, :, :, :, :] = r_vec
        return r_grid

    def create_er_orbs(self, n_rp: int, n_orbs: int) -> np.ndarray:
        orbs = np.zeros((n_rp, n_orbs, n_orbs, 2))
        for r, io1, io2 in it.product(range(n_rp), range(n_orbs), range(n_orbs)):
            orbs[r, io1, io2, :] = np.array([io1 + 1, io2 + 1])
        return orbs

    def create_ur_orbs(self, n_rp: int, n_orbs: int) -> np.ndarray:
        orbs = np.zeros((n_rp, n_orbs, n_orbs, n_orbs, n_orbs, 4))
        for r, io1, io2, io3, io4 in it.product(
            range(n_rp), range(n_orbs), range(n_orbs), range(n_orbs), range(n_orbs)
        ):
            orbs[r, io1, io2, io3, io4, :] = np.array([io1 + 1, io2 + 1, io3 + 1, io4 + 1])
        return orbs

    def insert_er_element(
        self, er_mat: np.ndarray, r2ind: dict[list, int], r_vec: list, orb1: int, orb2: int, hr_elem: float
    ) -> None:
        r_ind = r2ind[r_vec]
        er_mat[r_ind, orb1 - 1, orb2 - 1] = hr_elem

    def insert_ur_element(
        self,
        ur_mat: np.ndarray,
        r2ind: dict[list, int],
        r_vec: list,
        orb1: int,
        orb2: int,
        orb3: int,
        orb4: int,
        value: float,
    ) -> None:
        if r2ind is None or r_vec is None:
            ur_mat[orb1 - 1, orb2 - 1, orb3 - 1, orb4 - 1] = value
            return
        r_ind = r2ind[r_vec]
        ur_mat[r_ind, orb1 - 1, orb2 - 1, orb3 - 1, orb4 - 1] = value

    def parse_to_hopping_elements(self, hopping_elements: list) -> np.ndarray:
        return np.array([HoppingElement(**element) for element in hopping_elements])

    def parse_to_interaction_elements(self, interaction_elements: list) -> ndarray:
        return np.array([InteractionElement(**element) for element in interaction_elements])
