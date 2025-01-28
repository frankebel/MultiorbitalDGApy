import itertools as it
import logging

from mpi4py import MPI

import config
import dga_io
import local_sde
import plotting
from config_parser import ConfigParser
from dga_decorators import timeit
from local_greens_function import LocalGreensFunction
from memory_helper import MemoryHelper

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

comm = MPI.COMM_WORLD


@timeit
def execute_dga_routine():
    ConfigParser().parse_config(comm)

    g_dmft, sigma_dmft, g2_dens, g2_magn = dga_io.load_from_w2dyn_file_and_update_config()

    sigma_dmft.save(name="sigma_dmft", output_dir=config.output.output_path)

    if config.output.do_plotting:
        g2_dens.plot(omega=0, name=f"G2_dens", output_dir=config.output.output_path)
        g2_magn.plot(omega=0, name=f"G2_magn", output_dir=config.output.output_path)
        g2_magn.plot(omega=-10, name=f"G2_magn", output_dir=config.output.output_path)
        g2_magn.plot(omega=10, name=f"G2_magn", output_dir=config.output.output_path)

    ek = config.lattice.hamiltonian.get_ek(config.lattice.k_grid)
    g_loc = LocalGreensFunction.create_g_loc(sigma_dmft, ek)
    u_loc = config.lattice.hamiltonian.get_local_uq()
    # u_nonloc = config.hamiltonian.get_nonlocal_uq(config.q_grid)

    gamma_dens, gamma_magn, chi_dens, chi_magn, vrg_dens, vrg_magn, sigma = local_sde.perform_local_schwinger_dyson(
        g_loc, g2_dens, g2_magn, u_loc
    )

    if config.output.save_quantities:
        gamma_dens.save(name="Gamma_dens", output_dir=config.output.output_path)
        gamma_magn.save(name="Gamma_magn", output_dir=config.output.output_path)
        sigma.save(name="siw_sde_full", output_dir=config.output.output_path)
        chi_dens.save(name="chi_dens", output_dir=config.output.output_path)
        chi_magn.save(name="chi_magn", output_dir=config.output.output_path)
        vrg_dens.save(name="vrg_dens", output_dir=config.output.output_path)
        vrg_magn.save(name="vrg_magn", output_dir=config.output.output_path)

    if config.output.do_plotting:
        gamma_dens_plot = gamma_dens.cut_niv(min(config.box.niv, 2 * int(config.sys.beta)))
        gamma_dens_plot.plot(omega=0, name="Gamma_dens", output_dir=config.output.output_path)
        gamma_dens_plot.plot(omega=10, name="Gamma_dens", output_dir=config.output.output_path)
        gamma_dens_plot.plot(omega=-10, name="Gamma_dens", output_dir=config.output.output_path)
        MemoryHelper.delete(gamma_dens_plot)

        gamma_magn_plot = gamma_dens.cut_niv(min(config.box.niv, 2 * int(config.sys.beta)))
        gamma_magn_plot.plot(omega=0, name="Gamma_magn", output_dir=config.output.output_path)
        gamma_magn_plot.plot(omega=10, name="Gamma_magn", output_dir=config.output.output_path)
        gamma_magn_plot.plot(omega=-10, name="Gamma_magn", output_dir=config.output.output_path)
        MemoryHelper.delete(gamma_magn_plot)

        plotting.chi_checks(
            [chi_dens.mat], [chi_magn.mat], ["Loc-tilde"], g_loc, name="loc", output_dir=config.output.output_path
        )

        sigma_list = []
        sigma_names = []
        for i, j in it.product(range(config.sys.n_bands), repeat=2):
            try:
                sigma_list.append(sigma[i, j])
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
            name="1",
            output_dir=config.output.output_path,
        )

    print("Done!")


if __name__ == "__main__":
    execute_dga_routine()
