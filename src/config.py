import numpy as np

import brillouin_zone as bz
from hamiltonian import Hamiltonian

nk: tuple = (16, 16, 1)
nq: tuple = nk

niv: int = 60
niw: int = 60

beta: float = 0
mu: float = 0
n: float = 0
n_of_k: np.ndarray
u_dmft: float = 0

do_sym_v_vp = True

lattice_symmetry_set: str = "two_dimensional_square"
lattice_er_type: str = "t_tp_tpp"
lattice_er_input: list = [1, -0.2, 0.1]

input_path: str = "/home/julpe/Documents/DATA"
dmft_1p_filename: str = "1p-data.hdf5"
dmft_2p_filename: str = "g4iw_sym.hdf5"

k_grid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())
q_grid = bz.KGrid(nq, bz.two_dimensional_square_symmetries())

hamiltonian: Hamiltonian
