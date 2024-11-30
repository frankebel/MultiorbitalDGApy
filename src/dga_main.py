import logging

import numpy as np

import config
import dga_io
from hamiltonian import Hamiltonian, HamiltonianBuilder
from local_four_point import LocalFourPoint
from local_three_point import LocalThreePoint
from local_two_point import LocalTwoPoint
from local_self_energy import LocalSelfEnergy
from local_greens_function import LocalGreensFunction

logging.basicConfig(level=logging.DEBUG)

dmft_input = dga_io.load_from_w2dyn_file()
config.hamiltonian = (
    HamiltonianBuilder()
    .add_kinetic_one_band_2d_t_tp_tpp(*config.lattice_er_input)
    .add_single_band_interaction(dmft_input["u"])
    .build()
)

g2_dens = dmft_input["g4iw_dens"]
g2_magn = dmft_input["g4iw_magn"]

dga_io.update_frequency_boxes(g2_dens.niv, g2_dens.niw)

g2_dens = g2_dens.cut_niw_and_niv(config.niw, config.niv)
g2_magn = g2_magn.cut_niw_and_niv(config.niw, config.niv)
if config.do_sym_v_vp:
    g2_dens = g2_dens.symmetrize_v_vp()
    g2_magn = g2_magn.symmetrize_v_vp()

# g2_dens.plot(omega=0, figure_name=f"G2_dens_w{0}")
# g2_magn.plot(omega=0, figure_name=f"G2_magn_w{0}")
# g2_magn.plot(omega=-10, figure_name=f"G2_magn_w{-10}")
# g2_magn.plot(omega=10, figure_name=f"G2_magn_w{10}")

test = dmft_input["siw"].cut_niv(200)

greensFunction = LocalGreensFunction(
    np.empty_like(dmft_input["siw"].mat), dmft_input["siw"], config.hamiltonian.get_ek(config.k_grid)
)

print(dmft_input)
