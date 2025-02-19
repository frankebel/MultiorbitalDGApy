import mpi4py.MPI as MPI

import config
from four_point import FourPoint
from greens_function import GreensFunction
from local_four_point import LocalFourPoint
from matsubara_frequencies import *
from mpi_distributor import MpiDistributor
from n_point_base import Channel


def get_gchi_q(giwk: GreensFunction, q_list: np.ndarray) -> FourPoint:
    """
    Returns gchi0^{qk}_{lmm'l'} = -beta * G^{k}_{ll'} * G^{k-q}_{m'm}
    """
    wn = MFHelper.wn(config.box.niw, return_only_positive=True)
    iws, iws2 = np.array([MFHelper.get_frequency_shift(wn_i, FrequencyShift.MINUS) for wn_i in wn], dtype=int).T

    niv_asympt_range = np.arange(-config.box.niv_full, config.box.niv_full)

    gchi0_q = np.zeros(
        (len(q_list),) + (config.sys.n_bands,) * 4 + (len(wn), 2 * config.box.niv_full),
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

    return FourPoint(gchi0_q, Channel.NONE, config.lattice.nq, config.lattice.nk, 1, 0, 1, 1, False)


def calculate_self_energy_q(
    comm: MPI.Comm, giwk: GreensFunction, gamma_dens: LocalFourPoint, gamma_magn: LocalFourPoint
):
    logger = config.logger
    logger.log_info("Initializing MPI distributor.")
    mpi_distributor = MpiDistributor.create_distributor(ntasks=config.lattice.q_grid.nk_irr, comm=comm, name="Q")
    comm.barrier()
    my_q_list = config.lattice.q_grid.get_irrq_list()[mpi_distributor.my_slice]

    logger.log_info("Starting with non-local DGA routine.")
    giwk_full = giwk.get_g_full()

    # check why paul has an extra 1/beta in the frequency sums here
    gchi0_q = get_gchi_q(giwk_full, my_q_list)
    logger.log_info("Calculated gchi0_q.")
    logger.log_memory_usage("gchi0_q", gchi0_q, n_exists=1)
    chi0_q = gchi0_q.sum_over_fermionic_dimensions(config.sys.beta, axis=(-1,))
    logger.log_info("Calculated chi0_q.")
    gchi0_q_core = gchi0_q.cut_niv(config.box.niv)
    chi0_q_core = gchi0_q_core.sum_over_fermionic_dimensions(config.sys.beta, axis=(-1,))
    logger.log_info("Calculated chi0_q_core.")

    # comm.Allreduce(MPI.IN_PLACE, local_gchi0_q, op=MPI.SUM)
