import itertools as it
import logging
import os

import numpy as np
from mpi4py import MPI

import scdga.config as config
import scdga.dga_io as dga_io
import scdga.eliashberg_solver as eliashberg_solver
import scdga.local_sde as local_sde
import scdga.nonlocal_sde as nonlocal_sde
import scdga.plotting as plotting
from scdga.config_parser import ConfigParser
from scdga.debug_util import count_nonzero_orbital_entries
from scdga.greens_function import GreensFunction

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

    (
        config.lattice,
        config.box,
        config.output,
        config.sys,
        config.self_consistency,
        config.eliashberg,
        config.lambda_correction,
    ) = comm.bcast(
        (
            config.lattice,
            config.box,
            config.output,
            config.sys,
            config.self_consistency,
            config.eliashberg,
            config.lambda_correction,
        ),
        root=0,
    )

    if comm.rank == 0 and config.sys.n_bands != 1 and config.lambda_correction.perform_lambda_correction:
        raise ValueError(
            "Lambda correction is not available for multi-band systems. Please disable it in the config file."
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
        g_dmft.save(name="g_dmft", output_dir=config.output.output_path)
        logger.log_info("Saved sigma_dmft as numpy file.")

    if config.output.do_plotting and comm.rank == 0:
        for g2, name in [(g2_dens, "G2_dens"), (g2_magn, "G2_magn")]:
            for omega in [0, -10, 10]:
                plotting.plot_nu_nup(g2, omega=omega, name=name, output_dir=config.output.plotting_path)
        logger.log_info("Plotted g2 (dens) and g2 (magn).")

    ek = config.lattice.hamiltonian.get_ek(config.lattice.k_grid)
    # ek = config.lattice.hamiltonian.get_ek(config.lattice.k_grid)[..., 0, 0][..., None, None]
    g_loc = GreensFunction.create_g_loc(sigma_dmft.create_with_asympt_up_to_core(), ek)
    g_loc.save(output_dir=config.output.output_path, name="g_loc")
    # g_loc.mat = g_loc.mat[..., 0, 0, :][..., None, None, :]
    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.q_grid)

    logger.log_info("Preprocessing done.")
    logger.log_info("Starting local Schwinger-Dyson equation (SDE).")

    if comm.rank == 0:
        count_nonzero_orbital_entries(g2_dens, "g2_dens")
        count_nonzero_orbital_entries(g2_magn, "g2_magn")

    if comm.rank == 0:
        (gamma_d, gamma_m, chi_d, chi_m, vrg_d, vrg_m, f_d, f_m, gchi_d, gchi_m, sigma_loc) = (
            local_sde.perform_local_schwinger_dyson(g_loc, g2_dens, g2_magn, u_loc)
        )
    else:
        (gamma_d, gamma_m, chi_d, chi_m, vrg_d, vrg_m, f_d, f_m, gchi_d, gchi_m, sigma_loc) = (None,) * 11

    if comm.rank == 0:
        count_nonzero_orbital_entries(g2_dens, "g2_dens")
        count_nonzero_orbital_entries(g2_magn, "g2_magn")

        count_nonzero_orbital_entries(gchi_d, "gchi_dens")
        count_nonzero_orbital_entries(gchi_m, "gchi_magn")

        count_nonzero_orbital_entries(gamma_d, "gamma_dens")
        count_nonzero_orbital_entries(gamma_m, "gamma_magn")
        count_nonzero_orbital_entries(chi_d, "chi_dens")
        count_nonzero_orbital_entries(chi_m, "chi_magn")
        count_nonzero_orbital_entries(vrg_d, "vrg_dens")
        count_nonzero_orbital_entries(vrg_m, "vrg_magn")
        count_nonzero_orbital_entries(f_d, "f_dens")
        count_nonzero_orbital_entries(f_m, "f_magn")

        count_nonzero_orbital_entries(sigma_loc, "sigma_loc")
        count_nonzero_orbital_entries(sigma_dmft, "sigma_dmft")

        count_nonzero_orbital_entries(u_loc, "u_loc")

    # there is no need to broadcast the other quantities
    gamma_d = comm.bcast(gamma_d, root=0)
    gamma_m = comm.bcast(gamma_m, root=0)
    sigma_loc = comm.bcast(sigma_loc, root=0)

    logger.log_info("Local Schwinger-Dyson equation (SDE) done.")

    if comm.rank == 0:
        (f_d + 3 * f_m).save(name="f_1dens_3magn_loc", output_dir=config.output.output_path)
        logger.log_info(
            "Saved (f_dens_loc + 3 f_magn_loc) as numpy file, which is needed for the double-counting correction."
        )

    if (config.lambda_correction.perform_lambda_correction or config.output.save_quantities) and comm.rank == 0:
        chi_d.save(name="chi_dens_loc", output_dir=config.output.output_path)
        chi_m.save(name="chi_magn_loc", output_dir=config.output.output_path)

    if config.output.save_quantities and comm.rank == 0:
        g2_dens.save(name="g2_dens_loc", output_dir=config.output.output_path)
        g2_magn.save(name="g2_magn_loc", output_dir=config.output.output_path)
        del g2_dens, g2_magn

        gamma_d.save(name="gamma_dens_loc", output_dir=config.output.output_path)
        gamma_m.save(name="gamma_magn_loc", output_dir=config.output.output_path)
        sigma_loc.save(name="siw_dga_local", output_dir=config.output.output_path)
        vrg_d.save(name="vrg_dens_loc", output_dir=config.output.output_path)
        vrg_m.save(name="vrg_magn_loc", output_dir=config.output.output_path)
        del vrg_d, vrg_m

        gchi_d.save(name="gchi_dens_loc", output_dir=config.output.output_path)
        gchi_m.save(name="gchi_magn_loc", output_dir=config.output.output_path)
        f_d.save(name="f_dens_loc", output_dir=config.output.output_path)
        f_m.save(name="f_magn_loc", output_dir=config.output.output_path)
        logger.log_info("Saved all relevant quantities as numpy files.")

    if config.output.do_plotting and comm.rank == 0:
        plotting.plot_nu_nup(gchi_d, omega=0, name=f"Gchi_dens", output_dir=config.output.plotting_path)
        plotting.plot_nu_nup(gchi_m, omega=0, name=f"Gchi_magn", output_dir=config.output.plotting_path)
        logger.log_info(f"Local generalized susceptibilities dens & magn plotted.")
        del gchi_m, gchi_d

        gamma_dens_plot = gamma_d.cut_niv(min(config.box.niv_core, 2 * int(config.sys.beta)))
        plotting.plot_nu_nup(gamma_dens_plot, omega=0, name="Gamma_dens", output_dir=config.output.plotting_path)
        plotting.plot_nu_nup(gamma_dens_plot, omega=10, name="Gamma_dens", output_dir=config.output.plotting_path)
        plotting.plot_nu_nup(gamma_dens_plot, omega=-10, name="Gamma_dens", output_dir=config.output.plotting_path)
        logger.log_info("Plotted gamma (dens).")
        del gamma_dens_plot

        gamma_magn_plot = gamma_d.cut_niv(min(config.box.niv_core, 2 * int(config.sys.beta)))
        plotting.plot_nu_nup(gamma_magn_plot, omega=0, name="Gamma_magn", output_dir=config.output.plotting_path)
        plotting.plot_nu_nup(gamma_magn_plot, omega=10, name="Gamma_magn", output_dir=config.output.plotting_path)
        plotting.plot_nu_nup(gamma_magn_plot, omega=-10, name="Gamma_magn", output_dir=config.output.plotting_path)
        logger.log_info("Plotted gamma (magn).")
        del gamma_magn_plot

        plotting.chi_checks(
            [chi_d.mat],
            [chi_m.mat],
            ["Loc-tilde"],
            g_loc.e_kin,
            name="loc",
            output_dir=config.output.plotting_path,
        )
        del chi_d, chi_m
        logger.log_info("Plotted checks of the susceptibility.")

        sigma_list = []
        sigma_names = []
        for i, j in it.product(range(config.sys.n_bands), repeat=2):
            try:
                sigma_list.append(sigma_loc[0, 0, 0, i, j])
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
            output_dir=config.output.plotting_path,
        )
        logger.log_info("Plotted local self-energies for comparison.")
        logger.log_info("Finished plotting.")

    logger.log_info("Local DGA routine finished.")

    logger.log_info("Starting non-local ladder-DGA routine.")
    sigma_dga = nonlocal_sde.calculate_self_energy_q(
        comm, g_loc, gamma_d, gamma_m, u_loc, v_nonloc, sigma_dmft, sigma_loc
    )
    del gamma_d, gamma_m, sigma_dmft, sigma_loc
    logger.log_info("Non-local ladder-DGA routine finished.")

    giwk_dga = GreensFunction.get_g_full(sigma_dga, config.sys.mu, ek).cut_niv(
        config.box.niv_full + config.box.niw_core
    )

    if config.output.save_quantities and comm.rank == 0:
        sigma_dga.save(name=f"sigma_dga", output_dir=config.output.output_path)
        logger.log_info("Saved non-local self-energy as numpy file.")

        giwk_dga.save(name=f"giwk_dga", output_dir=config.output.output_path)
        logger.log_info("Saved non-local Green's function as numpy file.")

    if (config.poly_fitting.do_poly_fitting and not config.self_consistency.use_poly_fit) and comm.rank == 0:
        sigma_fit = sigma_dga.fit_polynomial(config.poly_fitting.n_fit, config.poly_fitting.o_fit, config.box.niv_core)
        sigma_fit.save(name=f"sigma_dga_fitted", output_dir=config.output.output_path)
        logger.log_info(f"Fitted polynomial of degree {config.poly_fitting.o_fit} to sigma.")
        logger.log_info("Saved fitted non-local self-energy as numpy file.")
        del sigma_fit

    if config.output.do_plotting and comm.rank == 0:
        kx, ky = config.lattice.k_grid.kx_shift, config.lattice.k_grid.ky_shift
        plotting.plot_two_point_kx_ky(sigma_dga, kx, ky, name="Sigma_dga_kz0", output_dir=config.output.plotting_path)
        logger.log_info("Plotted non-local self-energy as a function of kx and ky.")

        plotting.plot_two_point_kx_ky(giwk_dga, kx, ky, name="Giwk_dga_kz0", output_dir=config.output.plotting_path)
        logger.log_info("Plotted non-local Green's function as a function of kx and ky.")

    logger.log_info("DGA routine finished.")

    if config.eliashberg.perform_eliashberg:
        if not np.allclose(config.lattice.q_grid.nk, config.lattice.k_grid.nk):
            raise ValueError("Eliashberg equation can only be solved when nq = nk.")
        logger.log_info("Starting with Eliashberg equation.")
        lambdas_sing, lambdas_trip, gaps_sing, gaps_trip = eliashberg_solver.solve(
            giwk_dga, g_loc, u_loc, v_nonloc, comm
        )

        if config.output.save_quantities and comm.rank == 0:
            np.savetxt(
                os.path.join(config.output.eliashberg_path, "eigenvalues.txt"),
                [lambdas_sing.real, lambdas_trip.real],
                delimiter=",",
                fmt="%.9f",
            )

            for i in range(len(gaps_sing)):
                suffix = f"{i}" if i != len(gaps_sing) - 1 else "max"
                gaps_sing[i].save(name=f"gap_sing_{suffix}", output_dir=config.output.eliashberg_path)
                gaps_trip[i].save(name=f"gap_trip_{suffix}", output_dir=config.output.eliashberg_path)
            logger.log_info("Saved singlet and triplet gap functions to files.")

        if config.output.do_plotting and comm.rank == 0:
            kx, ky = config.lattice.k_grid.kx_shift_closed, config.lattice.k_grid.ky_shift_closed
            for i in range(len(gaps_sing)):
                suffix = f"{i}" if i != len(gaps_sing) - 1 else "max"
                plotting.plot_gap_function(
                    gaps_sing[i], kx, ky, name=f"gap_sing_{suffix}", output_dir=config.output.eliashberg_path
                )
                plotting.plot_gap_function(
                    gaps_trip[i], kx, ky, name=f"gap_trip_{suffix}", output_dir=config.output.eliashberg_path
                )
            logger.log_info("Plotted singlet and triplet gap functions.")

    logger.log_info("Exiting ...")
    MPI.Finalize()


if __name__ == "__main__":
    execute_dga_routine()
