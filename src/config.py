import numpy as np

import brillouin_zone as bz
from hamiltonian import Hamiltonian


class InteractionConfig:
    def __init__(self):
        self.udd: float = 0.0
        self.udp: float = 0.0
        self.upp: float = 0.0
        self.uppod: float = 0.0
        self.jdd: float = 0.0
        self.jdp: float = 0.0
        self.jpp: float = 0.0
        self.jppod: float = 0.0
        self.vdd: float = 0.0
        self.vpp: float = 0.0


do_plotting: bool = True
save_quantities: bool = True

nk: tuple = (64, 64, 1)
nq: tuple = nk

niw: int = -1
niv: int = -1

beta: float = 0.0
mu: float = 0.0
n: float = 0.0
rho_orbs: np.ndarray
n_dmft: float = 0.0
n_bands: int = 1

do_sym_v_vp: bool = True

lattice_symmetry_set: str = "two_dimensional_square"
lattice_er_type: str = "t_tp_tpp"
lattice_er_input: list = [1, -0.25, 0.12]

input_path: str = "/home/julpe/Documents/DATA/Singleorb-DATA/N490_B10_Nv40_U10"
output_path: str = "/home/julpe/Documents/repos/MultiorbitalDGApy"
dmft_1p_filename: str = "1p-data.hdf5"
dmft_2p_filename: str = "g4iw_sym.hdf5"

k_grid: bz.KGrid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())
q_grid: bz.KGrid = bz.KGrid(nq, bz.two_dimensional_square_symmetries())

hamiltonian: Hamiltonian

interaction: InteractionConfig = InteractionConfig()
