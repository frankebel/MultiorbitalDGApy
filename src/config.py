import numpy as np

import brillouin_zone as bz
from hamiltonian import Hamiltonian


class DgaConfig:
    def __init__(self):
        self.box = BoxConfig()
        self.lattice = LatticeConfig()
        self.dmft = DmftConfig()
        self.system = SystemConfig()
        self.output = OutputConfig()


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


class BoxConfig:
    def __init__(self):
        self.niw: int = -1
        self.niv: int = -1


class LatticeConfig:
    def __init__(self):
        self.symmetries: list[bz.KnownSymmetries] = bz.two_dimensional_square_symmetries()
        self.type: str = "t_tp_tpp"
        self.er_input: list = [1, -0.25, 0.12]
        self.nk: tuple = (16, 16, 1)
        self.nq: tuple = nk

        self.interaction: InteractionConfig = InteractionConfig()
        self.hamiltonian: Hamiltonian = Hamiltonian()
        self.k_grid: bz.KGrid = bz.KGrid(nk, self.symmetries)
        self.q_grid: bz.KGrid = bz.KGrid(nq, self.symmetries)


class DmftConfig:
    def __init__(self):
        self.input_path: str = "/."
        self.fname_1p: str = "1p-data.hdf5"
        self.fname_2p: str = "g4iw_sym.hdf5"


class SystemConfig:
    def __init__(self):
        self.beta: float = 0.0
        self.mu: float = 0.0
        self.n: float = 0.0
        self.n_bands: int = 1
        self.occ: np.ndarray


class OutputConfig:
    def __init__(self):
        self.do_plotting: bool = True
        self.save_quantities: bool = True


do_plotting: bool = True
save_quantities: bool = True

nk: tuple = (64, 64, 1)
nq: tuple = nk

niw: int = -1
niv: int = 70

beta: float = 0.0
mu: float = 0.0
n: float = 0.0
occ: np.ndarray
n_dmft: float = 0.0
n_bands: int = 1

do_sym_v_vp: bool = True

lattice_symmetry_set: str = "two_dimensional_square"
lattice_er_type: str = "t_tp_tpp"
lattice_er_input: list = [1, -0.25, 0.12]

# input_path: str = "/home/julpe/Documents/DATA/Singleorb-DATA/N490_B10_Nv40_U10"
input_path: str = "/home/julpe/Documents/DATA/Singleorb-DATA/N085_B12.5_Nv140_U8"
# input_path: str = "/home/julpe/Documents/DATA/Multiorb-DATA"
output_path: str = "/home/julpe/Documents/repos/MultiorbitalDGApy"
fname_1p: str = "1p-data.hdf5"
fname_2p: str = "g4iw_sym.hdf5"

k_grid: bz.KGrid = bz.KGrid(nk, bz.two_dimensional_square_symmetries())
q_grid: bz.KGrid = bz.KGrid(nq, bz.two_dimensional_square_symmetries())

hamiltonian: Hamiltonian

interaction: InteractionConfig = InteractionConfig()
