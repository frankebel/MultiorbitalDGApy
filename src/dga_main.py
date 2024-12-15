import logging

from mpi4py import MPI

import config
import dga_io
import local_sde
import plotting
from dga_decorators import timeit
from hamiltonian import Hamiltonian
from local_greens_function import LocalGreensFunction
from local_n_point import LocalNPoint

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


@timeit
def execute_dga_routine():
    config.comm = MPI.COMM_WORLD
    config.rank = config.comm.rank

    g_dmft, sigma_dmft, g2_dens, g2_magn = dga_io.load_from_w2dyn_file_and_update_config()

    sigma_dmft = sigma_dmft.extend_to_multi_orbital(LocalNPoint.from_constant(1, 0, sigma_dmft.niv, 2, 0, 1, 0.0), 3)

    config.hamiltonian = (
        Hamiltonian().kinetic_one_band_2d_t_tp_tpp(*config.lattice_er_input).single_band_interaction(config.u_dmft)
    )
    config.hamiltonian = (
        Hamiltonian()
        .read_er_w2k(filename="wannier_hr_test.dat")
        .single_band_interaction_as_multiband(config.u_dmft, num_bands=3)
    )

    sigma_dmft.save(name="sigma_dmft")

    dga_io.update_frequency_boxes(g2_dens.niv, g2_dens.niw)
    g2_dens, g2_magn = dga_io.update_g2_from_dmft(g2_dens, g2_magn)

    if config.do_plotting:
        g2_dens.plot(omega=0, name=f"G2_dens")
        g2_magn.plot(omega=0, name=f"G2_magn")
        g2_magn.plot(omega=-10, name=f"G2_magn")
        g2_magn.plot(omega=10, name=f"G2_magn")

    ek = config.hamiltonian.get_ek(config.k_grid)
    g_loc = LocalGreensFunction.create_g_loc(sigma_dmft, ek)
    u_loc = config.hamiltonian.get_local_uq()
    # u_nonloc = config.hamiltonian.get_nonlocal_uq(config.q_grid)

    gamma_dens, gamma_magn, chi_dens, chi_magn, vrg_dens, vrg_magn, sigma = local_sde.perform_schwinger_dyson(
        g_loc, g2_dens, g2_magn, u_loc
    )

    if config.save_quantities:
        gamma_dens.save(name="Gamma_dens")
        gamma_magn.save(name="Gamma_magn")
        sigma.save(name="siw_sde_full")
        chi_dens.save(name="chi_dens")
        chi_magn.save(name="chi_magn")

    if config.do_plotting:
        plotting.chi_checks([chi_dens.mat], [chi_magn.mat], ["Loc-tilde"], g_loc, name="loc")
        plotting.sigma_loc_checks(
            [sigma[0, 0], sigma[1, 0], sigma[0, 1], sigma[1, 1], sigma_dmft[0, 0]],
            ["SDE00", "SDE10", "SDE01", "SDE11", "Input"],
            config.beta,
            show=False,
            save=True,
            xmax=config.niv,
        )

    print("Done!")


if __name__ == "__main__":
    execute_dga_routine()
