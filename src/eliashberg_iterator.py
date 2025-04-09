import logging
import os

import mpi4py.MPI as MPI

import config
from four_point import FourPoint
from greens_function import GreensFunction
from interaction import LocalInteraction, Interaction
from n_point_base import SpinChannel
from mpi_distributor import MpiDistributor


def delete_files(filepath: str, filenames: list) -> None:
    for name in filenames:
        full_path = os.path.join(filepath, name)
        if os.path.isfile(full_path):
            try:
                os.remove(full_path)
            except OSError:
                config.logger.log_info(f"Error deleting file: {full_path}.")


def create_full_vertex_q(
    u_loc: LocalInteraction, v_nonloc: Interaction, channel: SpinChannel, comm: MPI.Comm
) -> FourPoint:
    # TODO: These objects are very large! Use Subcommunicator to reduce memory usage.

    logger = config.logger
    file_path = os.path.join(config.output.output_path, config.eliashberg.subfolder_name)

    logger.log_info("Loading gchi_aux_q_r, vrg_q_r, gchi_aux_q_r_sum, and gchi0_q_inv done.")

    u = u_loc.as_channel(channel) + v_nonloc.as_channel(channel)
    gchi0_q_inv = FourPoint.load(
        os.path.join(file_path, f"gchi0_q_inv_rank_{comm.rank}.npy"),
        num_vn_dimensions=1,
        has_compressed_q_dimension=True,
    )
    gchi_aux_q_r = FourPoint.load(
        os.path.join(file_path, f"gchi_aux_q_{channel.value}_rank_{comm.rank}.npy"),
        channel=channel,
        has_compressed_q_dimension=True,
    )
    f_q_r = config.sys.beta**2 * (gchi0_q_inv - gchi0_q_inv @ gchi_aux_q_r @ gchi0_q_inv)
    del gchi0_q_inv, gchi_aux_q_r

    delete_files(file_path, [f"gchi_aux_q_{channel.value}_rank_{comm.rank}.npy", f"gchi0_q_inv_rank_{comm.rank}.npy"])

    vrg_q_r = FourPoint.load(
        os.path.join(file_path, f"vrg_q_{channel.value}_rank_{comm.rank}.npy"),
        channel=channel,
        num_vn_dimensions=1,
        has_compressed_q_dimension=True,
    )
    gchi_aux_q_r_sum = FourPoint.load(
        os.path.join(file_path, f"gchi_aux_q_{channel.value}_sum_rank_{comm.rank}.npy"),
        channel=channel,
        num_vn_dimensions=0,
        has_compressed_q_dimension=True,
    )

    vrg_mul = vrg_q_r * vrg_q_r
    f_q_r += u @ vrg_mul - u @ gchi_aux_q_r_sum @ u @ vrg_mul
    del vrg_mul, vrg_q_r, gchi_aux_q_r_sum
    logger.log_info(f"Full momentum-dependent ladder-vertex ({f_q_r.channel.value}) calculated.")
    logger.log_memory_usage(f"Full vertex ({f_q_r.channel.value})", f_q_r, comm.size)

    delete_files(file_path, [f"vrg_q_{channel.value}_rank_{comm.rank}.npy", f"gchi0_q_inv_rank_{comm.rank}.npy"])

    f_q_r.save(
        name=f"f_q_{f_q_r.channel.value}_rank_{comm.rank}",
        output_dir=os.path.join(config.output.output_path, config.eliashberg.subfolder_name),
    )
    return f_q_r


def solve(giwk: GreensFunction, u_loc: LocalInteraction, v_nonloc: Interaction, comm: MPI.Comm):
    mpi_dist_irrk = MpiDistributor.create_distributor(ntasks=config.lattice.q_grid.nk_irr, comm=comm, name="Q")
    irrk_q_list = config.lattice.q_grid.get_irrq_list()
    my_irr_q_list = irrk_q_list[mpi_dist_irrk.my_slice]

    v_nonloc = v_nonloc.reduce_q(my_irr_q_list)

    # TODO: CHECK FREQUENCY NOTATION
    f_dens = create_full_vertex_q(u_loc, v_nonloc, SpinChannel.DENS, comm)
    f_magn = create_full_vertex_q(u_loc, v_nonloc, SpinChannel.MAGN, comm)
    f_sing = 0.5 * f_dens - 1.5 * f_magn
    f_sing.channel = SpinChannel.SING
    del f_dens
    f_sing = f_sing.change_frequency_notation_ph_to_pp()

    f_sing.save(
        name=f"f_q_{f_sing.channel.value}_rank_{comm.rank}",
        output_dir=os.path.join(config.output.output_path, config.eliashberg.subfolder_name),
    )

    f_trip = f_sing + 2 * f_magn
    f_trip.channel = SpinChannel.TRIP
    del f_magn
    f_trip = f_trip.change_frequency_notation_ph_to_pp()

    f_trip.save(
        name=f"f_q_{f_trip.channel.value}_rank_{comm.rank}",
        output_dir=os.path.join(config.output.output_path, config.eliashberg.subfolder_name),
    )
