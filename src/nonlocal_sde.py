import mpi4py.MPI as MPI

import config
from four_point import FourPoint
from greens_function import GreensFunction
from local_four_point import LocalFourPoint
from matsubara_frequencies import *
from mpi_distributor import MpiDistributor
from n_point_base import SpinChannel
from self_energy import SelfEnergy
from interaction import LocalInteraction, NonLocalInteraction


def create_generalized_chi0_q(giwk: GreensFunction, q_list: np.ndarray) -> FourPoint:
    """
    Returns gchi0^{qk}_{lmm'l'} = -beta * G^{k}_{ll'} * G^{k-q}_{m'm}
    """
    wn = MFHelper.wn(config.box.niw_core, return_only_positive=True)
    iws, iws2 = np.array([MFHelper.get_frequency_shift(wn_i, FrequencyShift.MINUS) for wn_i in wn], dtype=int).T

    niv_asympt_range = np.arange(-config.box.niv_full, config.box.niv_full)

    gchi0_q = np.zeros(
        (len(q_list),) + (config.sys.n_bands,) * 4 + (len(wn), 2 * config.box.niv_full),
        dtype=np.complex64,
    )

    g_left_mat = (
        giwk.mat[:, :, :, :, None, None, :, giwk.niv + niv_asympt_range[None, :] + iws[:, None]]
        * np.eye(config.sys.n_bands)[None, None, None, None, :, :, None, None, None]
    )

    for idx, q in enumerate(q_list):
        g_right_mat = (
            giwk.shift_k_by_q([-i for i in q]).transpose_orbitals()[
                :, :, :, None, :, :, None, giwk.niv + niv_asympt_range[None, :] + iws2[:, None]
            ]
            * np.eye(config.sys.n_bands)[None, None, None, :, None, None, :, None, None]
        )
        gchi0_q[idx] = -config.sys.beta * np.mean(g_left_mat * g_right_mat, axis=(0, 1, 2))

    return FourPoint(gchi0_q, SpinChannel.NONE, config.lattice.nq, 1, 1, full_niw_range=False)


def create_auxiliary_chi_q(
    gamma_r: LocalFourPoint, gchi0_q_inv: FourPoint, u_loc: LocalInteraction, v_nonloc: NonLocalInteraction
) -> FourPoint:
    return (
        (gchi0_q_inv + 1.0 / config.sys.beta**2 * gamma_r)
        - 1.0 / config.sys.beta**2 * (u_loc.as_channel(gamma_r.channel) + v_nonloc.as_channel(gamma_r.channel))
    ).invert()


def create_vrg_q(gchi_aux_q_r: FourPoint, gchi0_q_inv: FourPoint) -> FourPoint:
    gchi_aux_q_r_sum = gchi_aux_q_r.sum_over_vn(config.sys.beta, axis=(-1,))
    return config.sys.beta * (gchi0_q_inv @ gchi_aux_q_r_sum).take_vn_diagonal()


def create_generalized_chi_q_with_shell_correction(
    gchi_aux_q_sum: FourPoint,
    gchi0_q_full_sum: FourPoint,
    gchi0_q_core_sum: FourPoint,
    u_loc: LocalInteraction,
    v_nonloc: NonLocalInteraction,
) -> FourPoint:
    return (
        (gchi_aux_q_sum + gchi0_q_full_sum - gchi0_q_core_sum).invert()
        + u_loc.as_channel(gchi_aux_q_sum.channel)
        + v_nonloc.as_channel(gchi_aux_q_sum.channel)
    ).invert()


def create_vertex_functions_q(
    gamma_r: LocalFourPoint,
    gchi0_inv_core: FourPoint,
    gchi0_q_full_sum: FourPoint,
    gchi0_q_core_sum: FourPoint,
    u_loc: LocalInteraction,
    v_nonloc: NonLocalInteraction,
):
    logger = config.logger

    gchi_aux_q_r = create_auxiliary_chi_q(gamma_r, gchi0_inv_core, u_loc, v_nonloc)
    logger.log_info(f"Non-Local auxiliary susceptibility ({gchi_aux_q_r.channel.value}) calculated.")

    vrg_q_r = create_vrg_q(gchi_aux_q_r, gchi0_inv_core)
    logger.log_info(f"Non-local three-leg vertex gamma^wv ({vrg_q_r.channel.value}) done.")

    gchi_aux_q_r_sum = gchi_aux_q_r.sum_over_all_vn(config.sys.beta)
    del gchi_aux_q_r

    gchi_aux_q_r_sum = create_generalized_chi_q_with_shell_correction(
        gchi_aux_q_r_sum, gchi0_q_full_sum, gchi0_q_core_sum, u_loc, v_nonloc
    )
    logger.log_info(
        f"Updated non-local susceptibility chi^q ({gchi_aux_q_r_sum.channel.value}) with asymptotic correction."
    )

    return vrg_q_r, gchi_aux_q_r_sum


def get_hartree(u_loc: LocalInteraction, v_nonloc: NonLocalInteraction, q_list: np.ndarray) -> np.ndarray:
    v_q0 = v_nonloc.find_q((0, 0, 0))
    v_nonloc = v_nonloc.reduce_q(q_list)

    occ_qk = np.array([np.roll(config.sys.occ_k, [-i for i in q], axis=(0, 1, 2)) for q in q_list])  # [q,k,o1,o2]

    hartree_local_n = (u_loc.as_channel(SpinChannel.DENS) + v_q0).times("abcd,dc->ab", config.sys.occ)
    hartree_nonlocal_n = -v_nonloc.permute_orbitals()


def get_nonlocal_self_energy_vrg_q(
    vrg_dens_q: FourPoint,
    vrg_magn_q: FourPoint,
    gchi_dens_q_sum: FourPoint,
    gchi_magn_q_sum: FourPoint,
    giwk: GreensFunction,
    u_loc: LocalInteraction,
    v_nonloc: NonLocalInteraction,
    q_list: np.ndarray,
):
    g_qk = np.array(
        [
            MFHelper.wn_slices_gen(giwk.shift_k_by_q([-i for i in q]).mat, config.box.niv_core, config.box.niw_core)
            for q in q_list
        ]
    )  # has [q,k,o1,o2,w,v], should still be smaller than all other objects in size

    hartree_fock = get_hartree(u_loc, v_nonloc, q_list)


def calculate_self_energy_q(
    comm: MPI.Comm,
    giwk: GreensFunction,
    gamma_dens: LocalFourPoint,
    gamma_magn: LocalFourPoint,
    f_dens: LocalFourPoint,
    f_magn: LocalFourPoint,
    u_loc: LocalInteraction,
    v_nonloc: NonLocalInteraction,
) -> SelfEnergy:
    logger = config.logger
    logger.log_info("Initializing MPI distributor.")
    mpi_distributor = MpiDistributor.create_distributor(ntasks=config.lattice.q_grid.nk_irr, comm=comm, name="Q")
    comm.barrier()
    my_q_list = config.lattice.q_grid.get_irrq_list()[mpi_distributor.my_slice]

    logger.log_info("Starting with non-local DGA routine.")
    giwk_full = giwk.get_g_full()

    hartree = get_hartree(u_loc, v_nonloc, my_q_list)

    gchi0_q = create_generalized_chi0_q(giwk_full, my_q_list)  # this is for the q list of the current rank
    logger.log_info("Non-local bare susceptibility chi_0^qv done.")
    logger.log_memory_usage("gchi0_q", gchi0_q.memory_usage_in_gb, n_exists=1)

    gchi0_q_full_sum = 1.0 / config.sys.beta * gchi0_q.sum_over_all_vn(config.sys.beta)
    logger.log_info("Sum of chi_0^qv for the niv_full region done.")

    gchi0_q_core = gchi0_q.cut_niv(config.box.niv_core)
    del gchi0_q
    gchi_q_core_inv = gchi0_q_core.invert()
    logger.log_info("Inverted the non-local bare susceptibility chi_0^qv in the niv_core region.")

    gchi0_q_core_sum = 1.0 / config.sys.beta * gchi0_q_core.sum_over_all_vn(config.sys.beta)
    logger.log_info("Sum of chi_0^qv for the niv_core region done.")

    vrg_dens_q, gchi_dens_q_sum = create_vertex_functions_q(
        gamma_dens, gchi_q_core_inv, gchi0_q_full_sum, gchi0_q_core_sum, u_loc, v_nonloc
    )
    logger.log_info("Vertex functions for the density channel calculated.")
    vrg_magn_q, gchi_magn_q_sum = create_vertex_functions_q(
        gamma_magn, gchi_q_core_inv, gchi0_q_full_sum, gchi0_q_core_sum, u_loc, v_nonloc
    )
    logger.log_info("Vertex functions for the magnetic channel calculated.")
    del gamma_dens, gamma_magn, gchi_q_core_inv, gchi0_q_full_sum, gchi0_q_core_sum

    # comm.Allreduce(MPI.IN_PLACE, local_gchi0_q, op=MPI.SUM)
