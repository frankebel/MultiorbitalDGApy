import itertools as it
import os

import pandas as pd

from interaction import *


class HoppingElement:
    """
    Data class to store a hopping element of the Hamiltonian. The hopping element is defined by the relative lattice
    vector 'r_lat', the orbitals 'orbs' and the value of the hopping 'value'. A hopping element represents a single line
    in a w2k file. Added this to make the code more readable and to avoid passing around lists of values.
    """

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
    """
    Data class to store an interaction element of the Hamiltonian. The interaction element is defined by the relative lattice
    vector 'r_lat', the orbitals 'orbs' and the value of the interaction 'value'. An interaction element represents a single entry
    in the interaction matrix. Added this to make the code more readable and to avoid passing around lists of values.
    """

    def __init__(self, r_lat: list[int], orbs: list[int], value: float):
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
            raise ValueError("'value' must be a real number.")

        self.r_lat = tuple(r_lat)
        self.orbs = np.array(orbs, dtype=int)
        self.value = float(value)


class Hamiltonian:
    """
    Class to handle the Hamiltonian of the Hubbard model. Contains the hopping terms ('er') and the local ('local_interaction')
    and nonlocal interaction ('nonlocal_interaction') terms. Has a few helper methods to simplify adding terms to the Hamiltonian.
    """

    def __init__(self):
        self._er = None
        self._er_r_grid = None
        self._er_orbs = None
        self._er_r_weights = None

        self._local_interaction = None
        self._nonlocal_interaction = None

    def single_band_interaction(self, u: float) -> "Hamiltonian":
        """
        Sets the local interaction term for a single band model from the input of a floating point number.
        """
        interaction_elements = [InteractionElement([0, 0, 0], [1, 1, 1, 1], u)]
        return self._add_interaction_term(interaction_elements)

    def kinetic_one_band_2d_t_tp_tpp(self, t: float, tp: float, tpp: float) -> "Hamiltonian":
        """
        Adds the kinetic terms for a one band model in 2D with nearest, next-nearest and next-next-nearest neighbor hopping.
        """
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

        return self._add_kinetic_term(hopping_elements)

    def read_er_w2k(self, filepath: str = "./", filename: str = "wannier90.dat") -> "Hamiltonian":
        """
        Reads the 'wannier90.dat' file from the Wannier90 package and sets the Hamiltonian parameters accordingly.
        """
        hr_file = pd.read_csv(
            os.path.join(filepath, filename), skiprows=1, names=np.arange(15), sep=r"\s+", dtype=float, engine="python"
        )
        n_bands = hr_file.values[0][0].astype(int)
        nr = hr_file.values[1][0].astype(int)

        tmp = np.reshape(hr_file.values, (np.size(hr_file.values), 1))
        tmp = tmp[~np.isnan(tmp)]

        self._er_r_weights = tmp[2 : 2 + nr].astype(int)
        self._er_r_weights = np.reshape(self._er_r_weights, (np.size(self._er_r_weights), 1))
        ns = 7
        n_tmp = np.size(tmp[2 + nr :]) // ns
        tmp = np.reshape(tmp[2 + nr :], (n_tmp, ns))

        self._er_r_grid = np.reshape(tmp[:, 0:3], (nr, n_bands, n_bands, 3))
        self._er_orbs = np.reshape(tmp[:, 3:5], (nr, n_bands, n_bands, 2))
        self._er = np.reshape(tmp[:, 5] + 1j * tmp[:, 6], (nr, n_bands, n_bands))
        return self

    def write_hr_w2k(self, filename: str):
        """
        Write the real-space kinetic Hamiltonian in the format of w2k to a file.
        """
        n_columns = 15
        n_r = self._er.shape[0]
        n_bands = self._er.shape[-1]
        file = open(filename, "w", encoding="utf-8")
        file.write("# Written using the wannier module of the dga code\n")
        file.write(f"{n_bands} \n")
        file.write(f"{n_r} \n")

        for i in range(0, len(self._er_r_weights), n_columns):
            line = "    ".join(map(str, self._er_r_weights[i : i + n_columns, 0]))
            file.write("    " + line + "\n")
        hr = self._er.reshape(n_r * n_bands**2, 1)
        r_grid = self._er_r_grid.reshape(n_r * n_bands**2, 3).astype(int)
        orbs = self._er_orbs.reshape(n_r * n_bands**2, 2).astype(int)

        for i in range(0, n_r * n_bands**2):
            line = "{: 5d}{: 5d}{: 5d}{: 5d}{: 5d}{: 12.6f}{: 12.6f}".format(
                *r_grid[i, :], *orbs[i, :], hr[i, 0].real, hr[i, 0].imag
            )
            file.write(line + "\n")

    def get_ek(self, k_grid: bz.KGrid) -> np.ndarray:
        """
        Returns the band dispersion for the Hamiltonian on the input k-grid.
        """
        ek = self._convham_2_orbs(k_grid.kmesh.reshape(3, -1))
        n_orbs = ek.shape[-1]
        return ek.reshape(*k_grid.nk, n_orbs, n_orbs)

    def get_local_uq(self) -> LocalInteraction:
        """
        Returns the local interaction term in momentum space. Note, that due to the momentum-independence of the local interaction,
        the momentum space representation is the same as the real space representation.
        """
        return self._local_interaction

    def get_nonlocal_uq(self, q_grid: bz.KGrid) -> NonLocalInteraction:
        """
        Returns the nonlocal interaction term V^q in momentum space.
        """
        return self._nonlocal_interaction.get_uq(q_grid)

    def _add_kinetic_term(self, hopping_elements: list) -> "Hamiltonian":
        """
        Creates the kinetic term of the Hamiltonian from the hopping elements.
        """
        hopping_elements = self._parse_elements(hopping_elements, HoppingElement)

        if any(np.allclose(el.r_lat, [0, 0, 0]) for el in hopping_elements):
            raise ValueError("Local hopping is not allowed!")

        r_to_index, n_rp, n_orbs = self._prepare_indices_and_dimensions(hopping_elements)

        self._er_r_grid = self._create_er_grid(r_to_index, n_orbs)
        self._er_orbs = self._create_er_orbs(n_rp, n_orbs)
        self._er_r_weights = np.ones(n_rp)[:, None]

        self._er = np.zeros((n_rp, n_orbs, n_orbs))
        for he in hopping_elements:
            self._insert_er_element(self._er, r_to_index, he.r_lat, *he.orbs, he.value)
        return self

    def _add_interaction_term(self, interaction_elements: list) -> "Hamiltonian":
        """
        Creates the interaction term of the Hamiltonian from the interaction elements.
        """
        interaction_elements = self._parse_elements(interaction_elements, InteractionElement)
        r_to_index, n_rp, n_orbs = self._prepare_indices_and_dimensions(interaction_elements)

        ur_nonlocal_r_grid = self._create_ur_grid(r_to_index, n_orbs)
        ur_orbs = self._create_ur_orbs(n_rp, n_orbs)
        ur_nonlocal_r_weights = np.ones(n_rp)[:, None]

        ur_local = np.zeros((n_orbs, n_orbs, n_orbs, n_orbs))
        ur_nonlocal = np.zeros((n_rp, n_orbs, n_orbs, n_orbs, n_orbs))
        for ie in interaction_elements:
            if np.allclose(ie.r_lat, [0, 0, 0]):
                self._insert_ur_element(ur_local, None, None, *ie.orbs, ie.value)
            else:
                self._insert_ur_element(ur_nonlocal, r_to_index, ie.r_lat, *ie.orbs, ie.value)

        self._local_interaction = LocalInteraction(ur_local)
        self._nonlocal_interaction = NonLocalInteraction(ur_nonlocal, ur_nonlocal_r_grid, ur_nonlocal_r_weights)
        return self

    def _create_er_grid(self, r2ind: dict[list, int], n_orbs: int) -> np.ndarray:
        """
        Creates the real-space grid for the kinetic Hamiltonian.
        """
        n_rp = len(r2ind)
        r_grid = np.zeros((n_rp, n_orbs, n_orbs, 3))
        for r_vec in r2ind.keys():
            r_grid[r2ind[r_vec], :, :, :] = r_vec
        return r_grid

    def _create_ur_grid(self, r2ind: dict[list, int], n_orbs: int) -> np.ndarray:
        """
        Creates the real-space grid for the interaction Hamiltonian.
        """
        n_rp = len(r2ind)
        r_grid = np.zeros((n_rp, n_orbs, n_orbs, n_orbs, n_orbs, 3))
        for r_vec in r2ind.keys():
            r_grid[r2ind[r_vec], :, :, :, :, :] = r_vec
        return r_grid

    def _create_er_orbs(self, n_rp: int, n_orbs: int) -> np.ndarray:
        """
        Creates the orbital indices for the kinetic Hamiltonian.
        """
        orbs = np.zeros((n_rp, n_orbs, n_orbs, 2))
        for r, io1, io2 in it.product(range(n_rp), range(n_orbs), range(n_orbs)):
            orbs[r, io1, io2, :] = np.array([io1 + 1, io2 + 1])
        return orbs

    def _create_ur_orbs(self, n_rp: int, n_orbs: int) -> np.ndarray:
        """
        Creates the orbital indices for the interaction Hamiltonian.
        """
        orbs = np.zeros((n_rp, n_orbs, n_orbs, n_orbs, n_orbs, 4))
        for r, io1, io2, io3, io4 in it.product(
            range(n_rp), range(n_orbs), range(n_orbs), range(n_orbs), range(n_orbs)
        ):
            orbs[r, io1, io2, io3, io4, :] = np.array([io1 + 1, io2 + 1, io3 + 1, io4 + 1])
        return orbs

    def _insert_er_element(
        self, er_mat: np.ndarray, r_to_index: dict[list, int], r_vec: list, orb1: int, orb2: int, hr_elem: float
    ) -> None:
        """
        Inserts a hopping element into the kinetic Hamiltonian.
        """
        index = r_to_index[r_vec]
        er_mat[index, orb1 - 1, orb2 - 1] = hr_elem

    def _insert_ur_element(
        self,
        ur_mat: np.ndarray,
        r_to_index: dict[list, int] | None,
        r_vec: list | None,
        orb1: int,
        orb2: int,
        orb3: int,
        orb4: int,
        value: float,
    ) -> None:
        """
        Inserts an interaction element into the interaction Hamiltonian.
        """
        if r_to_index is None or r_vec is None:
            ur_mat[orb1 - 1, orb2 - 1, orb3 - 1, orb4 - 1] = value
            return
        index = r_to_index[r_vec]
        ur_mat[index, orb1 - 1, orb2 - 1, orb3 - 1, orb4 - 1] = value

    def _convham_2_orbs(self, k_mesh: np.ndarray = None) -> np.ndarray:
        """
        Fourier transforms the kinetic Hamiltonian to momentum space.
        """
        fft_grid = np.exp(1j * np.matmul(self._er_r_grid, k_mesh)) / self._er_r_weights[:, None, None]
        return np.transpose(np.sum(fft_grid * self._er[..., None], axis=0), axes=(2, 0, 1))

    def _parse_elements(self, elements: list, element_type: type) -> list:
        """
        Helper method. Parses the input elements to the correct data class.
        """
        if not all(isinstance(item, element_type) for item in elements):
            return [element_type(**element) for element in elements]
        return elements

    def _prepare_indices_and_dimensions(self, elements: list) -> tuple:
        """
        Helper method. Prepares the indices and dimensions for the Hamiltonian.
        """
        unique_lat_r = set([el.r_lat for el in elements])
        r_to_index = {tup: index for index, tup in enumerate(unique_lat_r)}
        n_rp = len(r_to_index)
        n_orbs = int(max(np.array([el.orbs for el in elements]).flatten()))
        return r_to_index, n_rp, n_orbs
