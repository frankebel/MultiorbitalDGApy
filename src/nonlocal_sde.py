import mpi4py.MPI as MPI

import config
from greens_function import GreensFunction
from matsubara_frequencies import *
from mpi_aux import MpiDistributor
from four_point import FourPoint
from n_point_base import Channel


def get_gchi_q_mat(giwk: GreensFunction, q_list: np.ndarray):
    """
    Returns -beta * G^k_ll' * G^(k-q)_m'm
    """
    wn = MFHelper.wn(config.box.niw)
    iws, iws2 = np.array([MFHelper.get_frequency_shift(wn_i, FrequencyShift.MINUS) for wn_i in wn], dtype=int).T

    niv_asympt_range = np.arange(-config.box.niv_full, config.box.niv_full)

    gchi0_q = np.zeros(
        (len(q_list),) + (config.sys.n_bands,) * 4 + (2 * config.box.niw + 1, 2 * config.box.niv_full),
        dtype=np.complex128,
    )

    g_left_mat = (
        giwk.mat[:, :, :, :, None, None, :, config.box.niv_full + niv_asympt_range[None, :] + iws[:, None]]
        * np.eye(config.sys.n_bands)[None, None, None, None, :, :, None, None, None]
    )

    for idx, q in enumerate(q_list):
        g_right_mat = (
            giwk.shift_k_by_q([-i for i in q]).mat.swapaxes(3, 4)[
                :, :, :, None, :, :, None, config.box.niv_full + niv_asympt_range[None, :] + iws2[:, None]
            ]
            * np.eye(config.sys.n_bands)[None, None, None, :, None, None, :, None, None]
        )
        gchi0_q[idx] = -config.sys.beta * np.mean(g_left_mat * g_right_mat, axis=(0, 1, 2))

    return FourPoint(gchi0_q, Channel.NONE, config.lattice.nq, config.lattice.nk, 1, 0, 1, 1)


def calculate_self_energy_q(comm: MPI.Comm, giwk: GreensFunction):
    logger = config.logger
    logger.log_info("Initializing MPI distributor.")
    mpi_distributor = MpiDistributor.create_distributor(ntasks=config.lattice.q_grid.nk_irr, comm=comm, name="Q")
    comm.barrier()
    my_q_list = config.lattice.q_grid.get_irrq_list()[mpi_distributor.my_slice]

    giwk_full = giwk.get_g_full()

    gchi0_q = get_gchi_q_mat(giwk_full, my_q_list)
    logger.log_info("Calculated gchi0_q.")

    # comm.Allreduce(MPI.IN_PLACE, local_gchi0_q, op=MPI.SUM)
