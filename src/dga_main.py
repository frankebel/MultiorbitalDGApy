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

comm = MPI.COMM_WORLD


def is_root() -> bool:
    return comm.rank == 0


def execute_dga_routine():
    ConfigParser().parse_config(comm)
    logger = config.logger
    logger.log_info("Starting DGA routine.")
    logger.log_info(f"Running on {str(comm.size)} {"process" if comm.size == 1 else "processes"}.")

    if is_root():
        g_dmft, sigma_dmft, g2_dens, g2_magn, g2_sing, g2_trip = dga_io.load_from_w2dyn_file_and_update_config()
    else:
        g_dmft, sigma_dmft, g2_dens, g2_magn, g2_sing, g2_trip = None, None, None, None, None, None

    logger.log_info("Config init and folder setup done.")
    logger.log_info("Loaded data from w2dyn file.")

    config.lattice, config.box, config.output, config.sys = comm.bcast(
        (config.lattice, config.box, config.output, config.sys), root=0
    )
    g_dmft, sigma_dmft, g2_dens, g2_magn, g2_sing, g2_trip = comm.bcast(
        (g_dmft, sigma_dmft, g2_dens, g2_magn, g2_sing, g2_trip), root=0
    )

    logger.log_memory_usage("giwk & siwk", g_dmft, 2)
    logger.log_memory_usage("g2_dens, g2_magn, g2_sing, g2_trip", g2_dens, 4)

    if config.output.save_quantities and is_root():
        sigma_dmft.save(name="sigma_dmft", output_dir=config.output.output_path)
        logger.log_info("Saved sigma_dmft as numpy file.")

    if config.output.do_plotting and is_root():
        for g2, name in [(g2_dens, "G2_dens"), (g2_magn, "G2_magn"), (g2_sing, "G2_sing"), (g2_trip, "G2_trip")]:
            for omega in [0, -10, 10]:
                g2.plot(omega=omega, name=name, output_dir=config.output.output_path)
        logger.log_info("Plotted G2 magn, dens, sing and trip.")

    ek = config.lattice.hamiltonian.get_ek(config.lattice.k_grid)
    g_loc = GreensFunction.create_g_loc(sigma_dmft, ek)
    u_loc = config.lattice.hamiltonian.get_local_uq()
    v_nonloc = config.lattice.hamiltonian.get_uq(config.lattice.q_grid)

    logger.log_info("Preprocessing done.")
    logger.log_info("Starting local Schwinger-Dyson equation (SDE).")
    (
        gchi_dens_loc,
        gchi_magn_loc,
        gchi0_full_loc,
        one_plus_gamma_dens_loc,
        one_plus_gamma_magn_loc,
        f_dens_loc,
        f_magn_loc,
        sigma_loc,
    ) = local_sde.perform_local_schwinger_dyson(g_loc, g2_dens, g2_magn, u_loc)
    logger.log_info("Local Schwinger-Dyson equation (SDE) done.")

    if config.output.save_quantities and is_root():
        gchi_dens_loc.save(name="gchi_dens_loc", output_dir=config.output.output_path)
        gchi_magn_loc.save(name="gchi_magn_loc", output_dir=config.output.output_path)
        f_dens_loc.save(name="f_dens_loc", output_dir=config.output.output_path)
        f_magn_loc.save(name="f_magn_loc", output_dir=config.output.output_path)
        one_plus_gamma_dens_loc.save(name="one_plus_gamma_dens_loc", output_dir=config.output.output_path)
        one_plus_gamma_magn_loc.save(name="one_plus_gamma_magn_loc", output_dir=config.output.output_path)
        gchi0_full_loc.save(name="gchi_0", output_dir=config.output.output_path)
        sigma_loc.save(name="siw_sde_full", output_dir=config.output.output_path)
        logger.log_info("Saved quantities as numpy files.")

    if config.output.do_plotting and is_root():
        gchi_dens_loc.plot(omega=0, name=f"Gchi_dens", output_dir=config.output.output_path)
        gchi_magn_loc.plot(omega=0, name=f"Gchi_magn", output_dir=config.output.output_path)
        logger.log_info("Generalized susceptibilities plotted.")

        sigma_list = []
        sigma_names = []
        for i, j in it.product(range(config.sys.n_bands), repeat=2):
            try:
                sigma_list.append(sigma_loc[i, j])
                sigma_list.append(sigma_dmft[i, j])
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
            xmax=config.box.niv,
            name="DMFT",
            output_dir=config.output.output_path,
        )
        logger.log_info("Plotted local self-energies for comparison.")

        logger.log_info("Finished plotting.")

    logger.log_info("Local DGA routine finished.")
    logger.log_info("Starting nonlocal ladder-DGA routine.")
    sigma_loc = nonlocal_sde.calculate_self_energy_q(
        comm,
        g_loc,
        gchi_dens_loc,
        gchi_magn_loc,
        one_plus_gamma_dens_loc,
        one_plus_gamma_magn_loc,
        f_dens_loc,
        f_magn_loc,
    )


if __name__ == "__main__":
    execute_dga_routine()
