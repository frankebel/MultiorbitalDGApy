import itertools as it
import logging
import os

import numpy as np
from mpi4py import MPI

import config
import dga_io
import eliashberg_solver
import local_sde
import nonlocal_sde
import plotting
from config_parser import ConfigParser
from greens_function import GreensFunction

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def execute_dga_routine():
    comm = MPI.COMM_WORLD

    config_parser = ConfigParser().parse_config(comm)
    logger = config.logger
    logger.log_info("Starting DGA routine.")
    logger.log_info(f"Running on {str(comm.size)} {"process" if comm.size == 1 else "processes"}.")

    if comm.rank == 0:
        g_dmft, sigma_dmft, g2_dens, g2_magn = dga_io.load_from_w2dyn_file_and_update_config()
    else:
        g_dmft, sigma_dmft, g2_dens, g2_magn = None, None, None, None

    config.lattice, config.box, config.output, config.sys, config.self_consistency = comm.bcast(
        (config.lattice, config.box, config.output, config.sys, config.self_consistency), root=0
    )

    config_parser.save_config_file(path=config.output.output_path, name="dga_config.yaml")

    logger.log_info("Config init and folder setup done.")
    logger.log_info("Loaded data from w2dyn file.")

    g_dmft = comm.bcast(g_dmft, root=0)
    sigma_dmft = comm.bcast(sigma_dmft, root=0)

    logger.log_memory_usage("giwk & siwk", g_dmft, 2 * comm.size)
    logger.log_memory_usage("g2_dens & g2_magn", g2_dens, 2 * comm.size)

    if config.output.save_quantities and comm.rank == 0:
        sigma_dmft.save(name="sigma_dmft", output_dir=config.output.output_path)
        logger.log_info("Saved sigma_dmft as numpy file.")

    if config.output.do_plotting and comm.rank == 0:
        for g2, name in [(g2_dens, "G2_dens"), (g2_magn, "G2_magn")]:
            for omega in [0, -10, 10]:
                g2.plot(omega=omega, name=name, output_dir=config.output.output_path)
        logger.log_info("Plotted g2 (dens) and g2 (magn).")

    ek = config.lattice.hamiltonian.get_ek(config.lattice.k_grid)
    giwk = GreensFunction.create_g_loc(sigma_dmft.create_with_asympt(), ek)
    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.q_grid)

    logger.log_info("Preprocessing done.")
    logger.log_info("Starting local Schwinger-Dyson equation (SDE).")

    if comm.rank == 0:
        gamma_dens, gamma_magn, chi_dens, chi_magn, vrg_dens, vrg_magn, f_dens, f_magn, sigma_local = (
            local_sde.perform_local_schwinger_dyson(giwk, g2_dens, g2_magn, u_loc)
        )
    else:
        gamma_dens, gamma_magn, chi_dens, chi_magn, vrg_dens, vrg_magn, f_dens, f_magn, sigma_local = (None,) * 9
    del g2_dens, g2_magn

    # there is no need to broadcast the other quantities
    gamma_dens = comm.bcast(gamma_dens, root=0)
    gamma_magn = comm.bcast(gamma_magn, root=0)
    sigma_local = comm.bcast(sigma_local, root=0)

    logger.log_info("Local Schwinger-Dyson equation (SDE) done.")

    if comm.rank == 0:
        (f_dens + 3 * f_magn).save(name="f_1dens_3magn_loc", output_dir=config.output.output_path)
        logger.log_info(
            "Saved [f_dens_loc + 3 f_magn_loc] as numpy file, which is needed for the double-counting correction."
        )

    if config.output.save_quantities and comm.rank == 0:
        gamma_dens.save(name="Gamma_dens", output_dir=config.output.output_path)
        gamma_magn.save(name="Gamma_magn", output_dir=config.output.output_path)
        sigma_local.save(name="siw_dga_local", output_dir=config.output.output_path)
        chi_dens.save(name="chi_dens", output_dir=config.output.output_path)
        chi_magn.save(name="chi_magn", output_dir=config.output.output_path)
        vrg_dens.save(name="vrg_dens", output_dir=config.output.output_path)
        vrg_magn.save(name="vrg_magn", output_dir=config.output.output_path)
        f_dens.save(name="f_dens_loc", output_dir=config.output.output_path)
        f_magn.save(name="f_magn_loc", output_dir=config.output.output_path)
        del vrg_dens, vrg_magn
        logger.log_info("Saved all relevant quantities as numpy files.")

    if config.eliashberg.perform_eliashberg and comm.rank == 0:
        (0.5 * f_dens - 0.5 * f_magn).save(name="f_ud_loc", output_dir=config.output.output_path)
        del f_dens, f_magn
        logger.log_info("Saved f_ud_loc as numpy file, which is needed for Gamma_pp.")

    if config.output.do_plotting and comm.rank == 0:
        gamma_dens_plot = gamma_dens.cut_niv(min(config.box.niv_core, 2 * int(config.sys.beta)))
        gamma_dens_plot.plot(omega=0, name="Gamma_dens", output_dir=config.output.output_path)
        gamma_dens_plot.plot(omega=10, name="Gamma_dens", output_dir=config.output.output_path)
        gamma_dens_plot.plot(omega=-10, name="Gamma_dens", output_dir=config.output.output_path)
        logger.log_info("Plotted gamma (dens).")
        del gamma_dens_plot

        gamma_magn_plot = gamma_dens.cut_niv(min(config.box.niv_core, 2 * int(config.sys.beta)))
        gamma_magn_plot.plot(omega=0, name="Gamma_magn", output_dir=config.output.output_path)
        gamma_magn_plot.plot(omega=10, name="Gamma_magn", output_dir=config.output.output_path)
        gamma_magn_plot.plot(omega=-10, name="Gamma_magn", output_dir=config.output.output_path)
        logger.log_info("Plotted gamma (magn).")
        del gamma_magn_plot

        plotting.chi_checks(
            [chi_dens.mat],
            [chi_magn.mat],
            ["Loc-tilde"],
            giwk,
            name="loc",
            output_dir=config.output.output_path,
        )
        del chi_dens, chi_magn
        logger.log_info("Plotted checks of the susceptibility.")

        sigma_list = []
        sigma_names = []
        for i, j in it.product(range(config.sys.n_bands), repeat=2):
            try:
                sigma_list.append(sigma_local[0, 0, 0, i, j])
                sigma_list.append(sigma_dmft[0, 0, 0, i, j])
                sigma_names.append(f"SDE{i}{j}")
                sigma_names.append(f"Input{i}{j}")
            except IndexError:
                break

        plotting.sigma_loc_checks(
            sigma_list,
            sigma_names,
            config.sys.beta,
            show=False,
            save=True,
            xmax=config.box.niv_core,
            name="DMFT",
            output_dir=config.output.output_path,
        )
        logger.log_info("Plotted local self-energies for comparison.")
        logger.log_info("Finished plotting.")

    logger.log_info("Local DGA routine finished.")

    logger.log_info("Starting non-local ladder-DGA routine.")
    sigma_dga = nonlocal_sde.calculate_self_energy_q(
        comm, giwk, gamma_dens, gamma_magn, u_loc, v_nonloc, sigma_dmft, sigma_local
    )
    logger.log_info("Non-local ladder-DGA routine finished.")

    if config.output.save_quantities and comm.rank == 0:
        sigma_dga.save(name=f"sigma_dga", output_dir=config.output.output_path)
        logger.log_info("Saved non-local self energy as numpy file.")

    if config.poly_fitting.do_poly_fitting and not config.self_consistency.use_poly_fit:
        sigma_fit = sigma_dga.fit_polynomial(config.poly_fitting.n_fit, config.poly_fitting.o_fit, config.box.niv_core)
        sigma_fit.save(name=f"sigma_dga_fitted", output_dir=config.output.output_path)
        logger.log_info(f"Fitted polynomial of degree {config.poly_fitting.o_fit} to sigma.")
        logger.log_info("Saved fitted non-local self energy as numpy file.")
        del sigma_fit

    del gamma_dens, gamma_magn, sigma_dmft, sigma_local

    logger.log_info("DGA routine finished.")

    if config.eliashberg.perform_eliashberg:
        if not np.allclose(config.lattice.q_grid.nk, config.lattice.k_grid.nk):
            raise ValueError("Eliashberg equation can only be solved when nq = nk.")
        logger.log_info("Starting with Eliashberg equation.")
        giwk = GreensFunction.get_g_full(sigma_dga, config.sys.mu, ek)
        lam_sing, lam_trip, gap_sing, gap_trip = eliashberg_solver.solve(giwk, u_loc, v_nonloc, comm)

        if config.output.save_quantities and comm.rank == 0:
            gap_sing.save(name=f"gap_sing", output_dir=config.output.eliashberg_path)
            gap_trip.save(name=f"gap_trip", output_dir=config.output.eliashberg_path)
            logger.log_info("Saved singlet and triplet gap functions to files.")

    logger.log_info("Exiting ...")
    MPI.Finalize()


if __name__ == "__main__":
    execute_dga_routine()
