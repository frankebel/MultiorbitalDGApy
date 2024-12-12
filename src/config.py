import numpy as np
from mpi4py.MPI import Intracomm

import brillouin_zone as bz
from hamiltonian import Hamiltonian

comm: Intracomm
rank: int = 0

do_plotting: bool = True
save_quantities: bool = True

nk: tuple = (26, 26, 1)
nq: tuple = nk

niw: int = 50
niv: int = 50

beta: float = 0.0
mu: float = 0.0
n: float = 0.0
rho_orbs: np.ndarray
n_dmft: float = 0.0
u_dmft: float = 0.0

do_sym_v_vp: bool = True

lattice_symmetry_set: str = "two_dimensional_square"
lattice_er_type: str = "t_tp_tpp"
lattice_er_input: list = [1, -0.25, 0.12]

input_path: str = "/home/julpe/Documents/DATA"
dmft_1p_filename: str = "1p-data.hdf5"
dmft_2p_filename: str = "g4iw_sym.hdf5"

k_grid: bz.KGrid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())
q_grid: bz.KGrid = bz.KGrid(nq, bz.two_dimensional_square_symmetries())

hamiltonian: Hamiltonian
