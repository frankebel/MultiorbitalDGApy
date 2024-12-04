import logging

import numpy as np
from mpi4py import MPI

import config
import dga_io
import local_sde
from dga_decorators import timeit
from hamiltonian import HamiltonianBuilder
from local_greens_function import LocalGreensFunction

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


@timeit
def execute_dga_routine():
    config.comm = MPI.COMM_WORLD
    config.rank = config.comm.rank

    g_loc, s_loc, g2_dens, g2_magn = dga_io.load_from_w2dyn_file_and_edit_config()
    config.hamiltonian = (
        HamiltonianBuilder()
        .add_kinetic_one_band_2d_t_tp_tpp(*config.lattice_er_input)
        .add_single_band_interaction(config.u_dmft)
        .build()
    )

    dga_io.update_frequency_boxes(g2_dens.niv, g2_dens.niw)
    g2_dens, g2_magn = dga_io.update_g2_from_dmft(g2_dens, g2_magn)

    if config.do_plotting:
        g2_dens.plot(omega=0, figure_name=f"G2_dens")
        g2_magn.plot(omega=0, figure_name=f"G2_magn")
        g2_magn.plot(omega=-10, figure_name=f"G2_magn")
        g2_magn.plot(omega=10, figure_name=f"G2_magn")

    ek = config.hamiltonian.kinetics.get_ek(config.k_grid)
    g_loc = LocalGreensFunction.create_g_loc(s_loc, ek)
    u_loc = config.hamiltonian.local_interaction

    sde = local_sde.calculate_local_self_energy(g_loc, g2_dens, g2_magn, u_loc)


if __name__ == "__main__":
    test = np.random.rand

    execute_dga_routine()
