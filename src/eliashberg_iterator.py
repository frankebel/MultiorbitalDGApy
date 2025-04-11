import os

import mpi4py.MPI as MPI
import numpy as np

import config
from four_point import FourPoint
from greens_function import GreensFunction
from interaction import LocalInteraction, Interaction
from matsubara_frequencies import MFHelper
from mpi_distributor import MpiDistributor
from n_point_base import SpinChannel, FrequencyNotation


def delete_files(filepath: str, *args) -> None:
    for name in args:
        if not isinstance(name, str):
            raise TypeError(f"Expected string, got {type(name)}.")
        full_path = os.path.join(filepath, name)
        if os.path.isfile(full_path):
            try:
                os.remove(full_path)
            except OSError:
                config.logger.log_info(f"Error deleting file: {name}.")


def gather_save_scatter(f_q_r: FourPoint, file_path: str, mpi_dist_irrk: MpiDistributor) -> FourPoint:
    if not config.output.save_quantities:
        return f_q_r

    f_q_r.mat = mpi_dist_irrk.gather(f_q_r.mat)
    if mpi_dist_irrk.my_rank == 0:
        f_q_r.save(output_dir=file_path, name=f"f_{f_q_r.channel.value}_irrq")
        config.logger.log_info(
            f"Saved full {f_q_r.channel.value} vertex {"in pp notation " if f_q_r.frequency_notation == FrequencyNotation.PP else ""}(for the irreducible BZ) to file."
        )
    f_q_r.mat = mpi_dist_irrk.scatter(f_q_r.mat)
    return f_q_r


def create_full_vertex_q_r(
    u_loc: LocalInteraction, v_nonloc: Interaction, channel: SpinChannel, comm: MPI.Comm
) -> FourPoint:
    logger = config.logger
    file_path = os.path.join(config.output.output_path, config.eliashberg.subfolder_name)

    logger.log_info(f"Starting to calculate the full {channel.value} vertex.")

    gchi0_q_inv = FourPoint.load(os.path.join(file_path, f"gchi0_q_inv_rank_{comm.rank}.npy"), num_vn_dimensions=1)
    gchi_aux_q_r = FourPoint.load(
        os.path.join(file_path, f"gchi_aux_q_{channel.value}_rank_{comm.rank}.npy"), channel=channel
    )
    logger.log_info(f"Loaded gchi0_q_inv and gchi_aux_q_{gchi_aux_q_r.channel.value} from files.")

    f_q_r = config.sys.beta**2 * (gchi0_q_inv - gchi0_q_inv @ gchi_aux_q_r @ gchi0_q_inv)
    del gchi0_q_inv, gchi_aux_q_r
    logger.log_info(f"Calculated first part of full {f_q_r.channel.value} vertex.")

    delete_files(file_path, f"gchi_aux_q_{channel.value}_rank_{comm.rank}.npy")

    vrg_q_r = FourPoint.load(
        os.path.join(file_path, f"vrg_q_{channel.value}_rank_{comm.rank}.npy"), channel=channel, num_vn_dimensions=1
    )
    gchi_aux_q_r_sum = FourPoint.load(
        os.path.join(file_path, f"gchi_aux_q_{channel.value}_sum_rank_{comm.rank}.npy"),
        channel=channel,
        num_vn_dimensions=0,
    )
    logger.log_info(
        f"Loaded vrg_q_{vrg_q_r.channel.value} and gchi_aux_q_{gchi_aux_q_r_sum.channel.value}_sum from files."
    )

    u = u_loc.as_channel(channel) + v_nonloc.as_channel(channel)
    u_vrg_mul = u @ (vrg_q_r * vrg_q_r)
    del vrg_q_r
    f_q_r += u_vrg_mul - u @ gchi_aux_q_r_sum @ u_vrg_mul
    del u, u_vrg_mul, gchi_aux_q_r_sum
    logger.log_info(f"Calculated second part of full {f_q_r.channel.value} vertex.")
    logger.log_info(f"Full momentum-dependent ladder-vertex ({f_q_r.channel.value}) calculated.")
    logger.log_memory_usage(f"Full vertex ({f_q_r.channel.value})", f_q_r, comm.size)

    delete_files(
        file_path, f"vrg_q_{channel.value}_rank_{comm.rank}.npy", f"gchi_aux_q_{channel.value}_sum_rank_{comm.rank}.npy"
    )

    return f_q_r


def create_full_vertex_pp_q_r(
    u_loc: LocalInteraction, v_nonloc: Interaction, channel: SpinChannel, mpi_dist_irrk: MpiDistributor
) -> FourPoint:
    file_path = os.path.join(config.output.output_path, config.eliashberg.subfolder_name)

    f_q_r = create_full_vertex_q_r(u_loc, v_nonloc, channel, mpi_dist_irrk.comm)
    f_q_r = gather_save_scatter(f_q_r, file_path, mpi_dist_irrk)

    niv_pp = min(config.box.niw_core // 2, config.box.niv_core // 2)
    vn = MFHelper.vn(niv_pp)
    omega = vn[:, None] - vn[None, :]

    f_q_r_flip = f_q_r.cut_niv(niv_pp).to_full_niw_range().flip_axis(-1)
    del f_q_r

    f_q_r_pp_mat = np.zeros((*f_q_r_flip.current_shape[:5], 2 * niv_pp, 2 * niv_pp), dtype=f_q_r_flip.mat.dtype)
    for idx, w in enumerate(MFHelper.wn(config.box.niw_core)):
        f_q_r_pp_mat[..., omega == w] = -f_q_r_flip[..., idx, omega == w]
    del f_q_r_flip

    config.logger.log_info(f"Calculated full {channel.value} vertex in pp notation.")
    return FourPoint(f_q_r_pp_mat, channel, config.lattice.q_grid.nk, 0, 2, True, True, True, FrequencyNotation.PP)


def solve(giwk: GreensFunction, u_loc: LocalInteraction, v_nonloc: Interaction, comm: MPI.Comm):
    logger = config.logger

    mpi_dist_irrk = MpiDistributor.create_distributor(ntasks=config.lattice.q_grid.nk_irr, comm=comm, name="Q")
    irrk_q_list = config.lattice.q_grid.get_irrq_list()
    my_irr_q_list = irrk_q_list[mpi_dist_irrk.my_slice]

    file_path = os.path.join(config.output.output_path, config.eliashberg.subfolder_name)

    v_nonloc = v_nonloc.reduce_q(my_irr_q_list)

    # TODO: These objects are very large! Use Subcommunicator to reduce memory usage.

    f_dens_pp = create_full_vertex_pp_q_r(u_loc, v_nonloc, SpinChannel.DENS, mpi_dist_irrk)
    f_magn_pp = create_full_vertex_pp_q_r(u_loc, v_nonloc, SpinChannel.MAGN, mpi_dist_irrk)

    delete_files(file_path, f"gchi0_q_inv_rank_{comm.rank}.npy")
    mpi_dist_irrk.delete_file()

    f_sing_pp = 0.5 * f_dens_pp - 1.5 * f_magn_pp
    f_sing_pp.channel = SpinChannel.SING
    del f_dens_pp

    f_sing_pp = gather_save_scatter(f_sing_pp, file_path, mpi_dist_irrk)

    f_trip_pp = f_sing_pp + 2 * f_magn_pp
    f_trip_pp.channel = SpinChannel.TRIP
    del f_magn_pp

    f_trip_pp = gather_save_scatter(f_trip_pp, file_path, mpi_dist_irrk)

    return 0, 0
