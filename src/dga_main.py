import logging
from copy import deepcopy

from mpi4py import MPI

import dga_io
import local_sde
import plotting
from config_parser import ConfigParser
from dga_decorators import timeit
from hamiltonian import Hamiltonian
from local_greens_function import LocalGreensFunction
from local_n_point import LocalNPoint
from memory_helper import MemoryHelper

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

comm = MPI.COMM_WORLD


@timeit
def execute_dga_routine():
    _ = ConfigParser().parse_config(comm)
    print("Hallo")
    exit()

    g_dmft, sigma_dmft, g2_dens, g2_magn = dga_io.load_from_w2dyn_file_and_update_config()

    sigma_dmft = sigma_dmft.extend_to_multi_orbital(LocalNPoint.from_constant(1, 0, sigma_dmft.niv, 2, 0, 1, 0.0), 2)

    # config.hamiltonian = (
    #    Hamiltonian()
    #    .kinetic_one_band_2d_t_tp_tpp(*config.lattice_er_input)
    #    .single_band_interaction(config.interaction.udd)
    # )
    config.hamiltonian = (
        Hamiltonian()
        .read_er_w2k(filename="/home/julpe/Documents/repos/MultiorbitalDGApy/wannier_hr.dat")
        .single_band_interaction_as_multiband(config.lattice.interaction.udd, 2)
    )

    # config.hamiltonian = (
    #    Hamiltonian()
    #    .read_er_w2k(filename="/home/julpe/Documents/repos/MultiorbitalDGApy/wannier_hr.dat")
    #    .kanamori_interaction(config.n_bands, config.interaction.udd, config.interaction.jdd, config.interaction.vdd)
    # )

    sigma_dmft.save(name="sigma_dmft")

    dga_io.update_frequency_boxes(g2_dens.niv, g2_dens.niw)
    g2_dens, g2_magn = dga_io.update_g2_from_dmft(g2_dens, g2_magn)

    if config.output.do_plotting:
        g2_dens.plot(omega=0, name=f"G2_dens")
        g2_magn.plot(omega=0, name=f"G2_magn")
        g2_magn.plot(omega=-10, name=f"G2_magn")
        g2_magn.plot(omega=10, name=f"G2_magn")

    ek = config.hamiltonian.get_ek(config.lattice.k_grid)
    g_loc = LocalGreensFunction.create_g_loc(sigma_dmft, ek)
    u_loc = config.hamiltonian.get_local_uq()
    # u_nonloc = config.hamiltonian.get_nonlocal_uq(config.q_grid)

    gamma_dens, gamma_magn, chi_dens, chi_magn, vrg_dens, vrg_magn, sigma = local_sde.perform_local_schwinger_dyson(
        g_loc, g2_dens, g2_magn, u_loc
    )

    if config.output.save_quantities:
        gamma_dens.save(name="Gamma_dens")
        gamma_magn.save(name="Gamma_magn")
        sigma.save(name="siw_sde_full")
        chi_dens.save(name="chi_dens")
        chi_magn.save(name="chi_magn")
        vrg_dens.save(name="vrg_dens")
        vrg_magn.save(name="vrg_magn")

    if config.output.do_plotting:
        gamma_dens_copy = deepcopy(gamma_dens)
        gamma_dens_copy = gamma_dens_copy.cut_niv(min(config.box.niv, 2 * int(config.sys.beta)))
        gamma_dens_copy.plot(omega=0, name="Gamma_dens")
        gamma_dens_copy.plot(omega=10, name="Gamma_dens")
        gamma_dens_copy.plot(omega=-10, name="Gamma_dens")
        MemoryHelper.delete(gamma_dens_copy)

        gamma_magn_copy = deepcopy(gamma_magn)
        gamma_magn_copy = gamma_magn_copy.cut_niv(min(config.box.niv, 2 * int(config.sys.beta)))
        gamma_magn_copy.plot(omega=0, name="Gamma_magn")
        gamma_magn_copy.plot(omega=10, name="Gamma_magn")
        gamma_magn_copy.plot(omega=-10, name="Gamma_magn")
        MemoryHelper.delete(gamma_magn_copy)

        plotting.chi_checks([chi_dens.mat], [chi_magn.mat], ["Loc-tilde"], g_loc, name="loc")
        plotting.sigma_loc_checks(
            [sigma[0, 0], sigma[1, 0], sigma[0, 1], sigma[1, 1], sigma_dmft[0, 0]],
            ["SDE00", "SDE10", "SDE01", "SDE11", "Input"],
            config.sys.beta,
            show=False,
            save=True,
            xmax=config.box.niv,
            name="1",
        )
        plotting.sigma_loc_checks(
            [sigma[0, 0], sigma_dmft[0, 0]],
            ["SDE00", "Input"],
            config.sys.beta,
            show=False,
            save=True,
            xmax=config.box.niv,
            name="2",
        )

    print("Done!")


if __name__ == "__main__":
    execute_dga_routine()
