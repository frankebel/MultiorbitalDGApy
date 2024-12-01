import logging

import numpy as np

import config
import dga_io
from hamiltonian import HamiltonianBuilder
from local_greens_function import LocalGreensFunction
from local_sde import calculate_local_self_energy

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def execute_dga_routine():
    g_loc, siw, g2_dens, g2_magn = dga_io.load_from_w2dyn_file_and_edit_config()
    config.hamiltonian = (
        HamiltonianBuilder()
        .add_kinetic_one_band_2d_t_tp_tpp(*config.lattice_er_input)
        .add_single_band_interaction(config.u_dmft)
        .build()
    )

    dga_io.update_frequency_boxes(g2_dens.niv, g2_dens.niw)
    g2_dens, g2_magn = dga_io.update_g2_from_dmft(g2_dens, g2_magn)

    # g2_dens.plot(omega=0, figure_name=f"G2_dens_w{0}")
    # g2_magn.plot(omega=0, figure_name=f"G2_magn_w{0}")
    # g2_magn.plot(omega=-10, figure_name=f"G2_magn_w{-10}")
    # g2_magn.plot(omega=10, figure_name=f"G2_magn_w{10}")

    g_loc = LocalGreensFunction.create_from_sigma_ek(siw, config.hamiltonian.get_ek(config.k_grid))

    sde = calculate_local_self_energy(g_loc, g2_dens, g2_magn)


if __name__ == "__main__":
    execute_dga_routine()
