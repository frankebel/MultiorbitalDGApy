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

    return FourPoint(
        gchi0_q, SpinChannel.NONE, config.lattice.nq, 1, 1, full_niw_range=False, has_compressed_momentum_dimension=True
    )


def create_auxiliary_chi_q(
    gamma_r: LocalFourPoint, gchi0_q_inv: FourPoint, u_loc: LocalInteraction, v_nonloc: NonLocalInteraction
) -> FourPoint:
    r"""
    Returns the auxiliary susceptibility
    .. math:: \chi^{*;qvv'}_{r;lmm'l'} = ((\chi_{0;lmm'l'}^{qv})^{-1} + (\Gamma_{r;lmm'l'}^{qvv'}-U_{r;lmm'l'}-V_{r;lmm'l'}^q)/\beta^2)^{-1}

    .. math:: = ((\chi_{0;lmm'l'}^{qv})^{-1} + (\Gamma_{r;lmm'l'}^{wvv'}-U_{r;lmm'l'})/\beta^2)^{-1}.
    See Eq. (3.68) in Paul Worm's thesis.
    """
    return (
        (gchi0_q_inv + 1.0 / config.sys.beta**2 * gamma_r)
        - 1.0 / config.sys.beta**2 * (v_nonloc.as_channel(gamma_r.channel) + u_loc.as_channel(gamma_r.channel))
    ).invert()


def create_generalized_chi_q_with_shell_correction(
    gchi_aux_q_sum: FourPoint,
    gchi0_q_full_sum: FourPoint,
    gchi0_q_core_sum: FourPoint,
    u_loc: LocalInteraction,
    v_nonloc: NonLocalInteraction,
) -> FourPoint:
    """
    Calculates the generalized susceptibility with the shell correction as described by
    Motoharu Kitatani et al. 2022 J. Phys. Mater. 5 034005; DOI 10.1088/2515-7639/ac7e6d. Eq. A.15
    """
    return (
        (gchi_aux_q_sum + gchi0_q_full_sum - gchi0_q_core_sum).invert()
        + (u_loc.as_channel(gchi_aux_q_sum.channel) + v_nonloc.as_channel(gchi_aux_q_sum.channel))
    ).invert()


def create_vrg_q(gchi_aux_q_r: FourPoint, gchi0_q_inv: FourPoint) -> FourPoint:
    r"""
    Returns the three-leg vertex
    .. math:: \gamma_{r;lmm'l'}^{qv} = \beta * (\chi^{qvv}_{0;lmab})^{-1} * (\sum_{v'} \chi^{*;qvv'}_{r;bam'l'}).
    See Eq. (3.71) in Paul Worm's thesis.
    """
    gchi_aux_q_r_sum = gchi_aux_q_r.sum_over_vn(config.sys.beta, axis=(-1,))
    return config.sys.beta * (gchi0_q_inv @ gchi_aux_q_r_sum).take_vn_diagonal()


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
    del gamma_r
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


def get_hartree_fock(u_loc: LocalInteraction, v_nonloc: NonLocalInteraction, full_q_list: np.ndarray) -> np.ndarray:
    r"""
    Returns the Hartree-Fock term for the local and non-local interaction.
    .. math:: \Sigma_{HF}^k = (2*U_{abcd} - U_{adcb} + 2*V^{q=0}_{abcd}) n_{dc} - 1/N_q \sum_q V^{q}_{adcb} n^{k-q}_{dc}
    where

    .. math:: 2*U_{abcd} - U_{adcb} = U_{dens,abcd}.
    """
    v_q0 = v_nonloc.find_q((0, 0, 0))
    occ_qk = np.array([np.roll(config.sys.occ_k, [-i for i in q], axis=(0, 1, 2)) for q in full_q_list])  # [q,k,o1,o2]
    nq_tot, nk_tot = np.prod(config.lattice.nq), np.prod(config.lattice.nk)
    occ_qk = occ_qk.reshape(nq_tot, nk_tot, config.sys.n_bands, config.sys.n_bands)

    hartree_local_n = (u_loc.as_channel(SpinChannel.DENS) + 2 * v_q0).times("qabcd,dc->ab", config.sys.occ)
    hartree_nonlocal_n = (
        -1.0 / nq_tot * v_nonloc.compress_q_dimension().permute_orbitals("abcd->adcb").times("qabcd,qkdc->kab", occ_qk)
    )
    return hartree_local_n[None, ...] + hartree_nonlocal_n


def get_sigma_kernel_vrg_r_q(
    vrg_r_q: FourPoint, gchi_r_q_sum: FourPoint, u_loc: LocalInteraction, v_nonloc: NonLocalInteraction
) -> tuple[NonLocalInteraction, FourPoint]:
    r"""
    Returns the kernel for the self-energy calculation.
    .. math:: K = \gamma_{r;abcd}^{qv} - \gamma_{r;abef}^{qv} * U^{q}_{r;fehg} * \chi_{r;ghcd}^{q}
    """
    u_r = v_nonloc.as_channel(vrg_r_q.channel) + u_loc.as_channel(vrg_r_q.channel)
    return u_r, vrg_r_q - vrg_r_q @ u_r @ gchi_r_q_sum


def calculate_u_kernel_g_mat(u_r: NonLocalInteraction, kernel_r: FourPoint, g_qk: np.ndarray) -> np.ndarray:
    r"""
    Returns
    .. math:: \Sigma_{ij}^{k} = -1/\beta \sum_q [ U^q_{aibc} * K_{cbjd}^{qv} * G_{ad}^{w-v} ],
    where

    .. math:: K_{r;abcd}^{qv} = \gamma_{r;abcd}^{qv} - \gamma_{r;abef}^{qv} * U^{q}_{r;fehg} * \chi_{r;ghcd}^{q}
    """
    return -1.0 / config.sys.beta / np.prod(config.lattice.nq) * u_r.times("qrjop,qilpowv,qklrwv->kijv", kernel_r, g_qk)


def get_nonlocal_self_energy_q(
    u_dens: NonLocalInteraction,
    u_magn: NonLocalInteraction,
    kernel_dens: FourPoint,
    kernel_magn: FourPoint,
    g_qk: np.ndarray,
):
    """
    Returns the non-local self-energy from kernel functions. See big [] brackets in my thesis Eq. (1.125).
    """
    return calculate_u_kernel_g_mat(u_dens, kernel_dens, g_qk) + 3 * calculate_u_kernel_g_mat(u_magn, kernel_magn, g_qk)


def get_self_energy_dc_kernel_q(
    f_dens: LocalFourPoint,
    f_magn: LocalFourPoint,
    f_dens_2: LocalFourPoint,
    f_magn_2: LocalFourPoint,
    u_loc: LocalInteraction,
    gchi0_q_core: FourPoint,
    g_qk: np.ndarray,
):
    kernel_dens_1 = (gchi0_q_core @ f_dens).sum_over_vn(config.sys.beta, axis=(-2,))
    kernel_magn_1 = (gchi0_q_core @ f_magn).sum_over_vn(config.sys.beta, axis=(-2,))
    kernel_dens_2 = (gchi0_q_core @ f_dens_2).sum_over_vn(config.sys.beta, axis=(-2,))
    kernel_magn_2 = (gchi0_q_core @ f_magn_2).sum_over_vn(config.sys.beta, axis=(-2,))

    def calc_dc(kernel_r):
        return (
            0.5
            / config.sys.beta
            / np.prod(config.lattice.nq)
            * u_loc.as_channel(SpinChannel.MAGN).times("rjop,qilpowv,qklrwv->kijv", kernel_r, g_qk)
        )

    sigma_dc_dens_1 = calc_dc(kernel_dens_1)
    sigma_dc_dens_2 = calc_dc(kernel_dens_2)
    sigma_dc_magn_1 = calc_dc(kernel_magn_1)
    sigma_dc_magn_2 = calc_dc(kernel_magn_2)

    np.save("/home/julpe/Desktop/sigma_dc_dens_1.npy", sigma_dc_dens_1)
    np.save("/home/julpe/Desktop/sigma_dc_dens_2.npy", sigma_dc_dens_2)
    np.save("/home/julpe/Desktop/sigma_dc_magn_1.npy", sigma_dc_magn_1)
    np.save("/home/julpe/Desktop/sigma_dc_magn_2.npy", sigma_dc_magn_2)

    return sigma_dc_dens_1, sigma_dc_dens_2, sigma_dc_magn_1, sigma_dc_magn_2


def get_g_qk_wv(giwk: GreensFunction, q_list: np.ndarray) -> np.ndarray:
    g_qk = np.array(
        [
            MFHelper.wn_slices_gen(giwk.shift_k_by_q([-i for i in q]).mat, config.box.niv_core, config.box.niw_core)
            for q in q_list
        ]
    ).reshape(
        len(q_list),
        np.prod(config.lattice.nk),
        config.sys.n_bands,
        config.sys.n_bands,
        2 * config.box.niw_core + 1,
        2 * config.box.niv_core,
    )  # has [q,k,o1,o2,w,v], should be atleast (#o)^2 times smaller than most other objects in size
    return g_qk


def calculate_self_energy_q(
    comm: MPI.Comm,
    giwk: GreensFunction,
    gamma_dens: LocalFourPoint,
    gamma_magn: LocalFourPoint,
    f_dens: LocalFourPoint,
    f_magn: LocalFourPoint,
    u_loc: LocalInteraction,
    v_nonloc: NonLocalInteraction,
    f_dens_2: LocalFourPoint,
    f_magn_2: LocalFourPoint,
) -> SelfEnergy:
    logger = config.logger
    logger.log_info("Initializing MPI distributor.")
    mpi_distributor = MpiDistributor.create_distributor(ntasks=config.lattice.q_grid.nk_irr, comm=comm, name="Q")
    comm.barrier()
    full_q_list = config.lattice.q_grid.get_q_list()
    my_q_list = config.lattice.q_grid.get_irrq_list()[mpi_distributor.my_slice]
    my_full_q_list = config.lattice.q_grid.get_q_list()[mpi_distributor.my_slice]

    hartree = get_hartree_fock(u_loc, v_nonloc, full_q_list)
    v_nonloc = v_nonloc.reduce_q(my_q_list)

    logger.log_info("Starting with non-local DGA routine.")
    giwk_full = giwk.get_g_full()
    del giwk

    gchi0_q = create_generalized_chi0_q(giwk_full, my_q_list)  # this is for the q list of the current rank
    logger.log_info("Non-local bare susceptibility chi_0^qv done.")
    logger.log_memory_usage("gchi0_q", gchi0_q.memory_usage_in_gb, n_exists=1)

    gchi0_q_full_sum = 1.0 / config.sys.beta * gchi0_q.sum_over_all_vn(config.sys.beta)
    logger.log_info("Sum of chi_0^qv for the niv_full region done.")

    gchi0_q_core = gchi0_q.cut_niv(config.box.niv_core)
    del gchi0_q
    gchi_q_core_inv = gchi0_q_core.invert().take_vn_diagonal()
    logger.log_info("Inverted the non-local bare susceptibility chi_0^qv in the niv_core region.")

    g_qk = get_g_qk_wv(giwk_full, my_q_list)
    sigma_dc_dens_1, sigma_dc_dens_2, sigma_dc_magn_1, sigma_dc_magn_2 = get_self_energy_dc_kernel_q(
        f_dens, f_magn, f_dens_2, f_magn_2, u_loc, gchi0_q_core, g_qk
    )
    del sigma_dc_dens_1, sigma_dc_dens_2, sigma_dc_magn_1, sigma_dc_magn_2, f_dens, f_magn, f_dens_2, f_magn_2

    gchi0_q_core_sum = 1.0 / config.sys.beta * gchi0_q_core.sum_over_all_vn(config.sys.beta)
    del gchi0_q_core
    logger.log_info("Sum of chi_0^qv for the niv_core region done.")

    vrg_dens_q, gchi_dens_q_sum = create_vertex_functions_q(
        gamma_dens, gchi_q_core_inv, gchi0_q_full_sum, gchi0_q_core_sum, u_loc, v_nonloc
    )
    del gamma_dens
    logger.log_info("Vertex functions for the density channel calculated.")

    u_dens, kernel_dens = get_sigma_kernel_vrg_r_q(vrg_dens_q, gchi_dens_q_sum, u_loc, v_nonloc)
    del vrg_dens_q, gchi_dens_q_sum
    logger.log_info("Kernel function for sigma for the density channel calculated.")

    vrg_magn_q, gchi_magn_q_sum = create_vertex_functions_q(
        gamma_magn, gchi_q_core_inv, gchi0_q_full_sum, gchi0_q_core_sum, u_loc, v_nonloc
    )
    del gamma_magn, gchi_q_core_inv, gchi0_q_full_sum, gchi0_q_core_sum
    logger.log_info("Vertex functions for the magnetic channel calculated.")

    u_magn, kernel_magn = get_sigma_kernel_vrg_r_q(vrg_magn_q, gchi_magn_q_sum, u_loc, v_nonloc)
    del vrg_magn_q, gchi_magn_q_sum
    logger.log_info("Kernel function for sigma for the magnetic channel calculated.")

    sigma = get_nonlocal_self_energy_q(u_dens, u_magn, kernel_dens, kernel_magn, g_qk)

    # comm.Allreduce(MPI.IN_PLACE, local_gchi0_q, op=MPI.SUM)
