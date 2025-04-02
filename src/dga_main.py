import itertools as it
import logging

from mpi4py import MPI

import config
import dga_io
import local_sde
import nonlocal_sde
import plotting
from config_parser import ConfigParser
from greens_function import GreensFunction

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def execute_dga_routine():
    comm = MPI.COMM_WORLD

    ConfigParser().parse_config(comm)
    logger = config.logger
    logger.log_info("Starting DGA routine.")
    logger.log_info(f"Running on {str(comm.size)} {"process" if comm.size == 1 else "processes"}.")

    if comm.rank == 0:
        g_dmft, sigma_dmft, g2_dens, g2_magn = dga_io.load_from_w2dyn_file_and_update_config()
    else:
        g_dmft, sigma_dmft, g2_dens, g2_magn = None, None, None, None

    logger.log_info("Config init and folder setup done.")
    logger.log_info("Loaded data from w2dyn file.")

    config.lattice, config.box, config.output, config.sys, config.self_consistency = comm.bcast(
        (config.lattice, config.box, config.output, config.sys, config.self_consistency), root=0
    )

    g_dmft, sigma_dmft, g2_dens, g2_magn = comm.bcast((g_dmft, sigma_dmft, g2_dens, g2_magn), root=0)

    logger.log_memory_usage("giwk & siwk", g_dmft.memory_usage_in_gb, 2)
    logger.log_memory_usage("g2_dens & g2_magn", g2_dens.memory_usage_in_gb, 2)

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
        gamma_dens, gamma_magn, chi_dens, chi_magn, vrg_dens, vrg_magn, f_dens, f_magn, sigma_local = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    gamma_dens, gamma_magn, chi_dens, chi_magn, vrg_dens, vrg_magn, f_dens, f_magn, sigma_local = comm.bcast(
        (gamma_dens, gamma_magn, chi_dens, chi_magn, vrg_dens, vrg_magn, f_dens, f_magn, sigma_local)
    )
    logger.log_info("Local Schwinger-Dyson equation (SDE) done.")

    if config.output.save_quantities and comm.rank == 0:
        gamma_dens.save(name="Gamma_dens", output_dir=config.output.output_path)
        gamma_magn.save(name="Gamma_magn", output_dir=config.output.output_path)
        sigma_local.save(name="siw_dga_local", output_dir=config.output.output_path)
        chi_dens.save(name="chi_dens", output_dir=config.output.output_path)
        chi_magn.save(name="chi_magn", output_dir=config.output.output_path)
        vrg_dens.save(name="vrg_dens", output_dir=config.output.output_path)
        vrg_magn.save(name="vrg_magn", output_dir=config.output.output_path)
        f_dens.save(name="f_dens", output_dir=config.output.output_path)
        f_magn.save(name="f_magn", output_dir=config.output.output_path)
        logger.log_info("Saved all relevant quantities as numpy files.")

    del vrg_dens, vrg_magn, f_dens, f_magn

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

    if comm.rank == 0 and config.output.save_quantities:
        sigma_dga.save(name="sigma_dga", output_dir=config.output.output_path)
        logger.log_info("Saved sigma_dga as numpy file.")

    logger.log_info("DGA routine finished.")
    logger.log_info("Exiting ...")
    MPI.Finalize()


if __name__ == "__main__":
    execute_dga_routine()
