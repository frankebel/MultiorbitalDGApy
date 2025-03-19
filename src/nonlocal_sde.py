import mpi4py.MPI as MPI

import config
from four_point import FourPoint
from greens_function import GreensFunction
from interaction import LocalInteraction, Interaction
from local_four_point import LocalFourPoint
from matsubara_frequencies import *
from mpi_distributor import MpiDistributor
from n_point_base import SpinChannel
from self_energy import SelfEnergy


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
    gamma_r: LocalFourPoint, gchi0_q_inv: FourPoint, u_loc: LocalInteraction, v_nonloc: Interaction
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
    v_nonloc: Interaction,
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
    v_nonloc: Interaction,
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


def get_hartree_fock(
    u_loc: LocalInteraction, v_nonloc: Interaction, full_q_list: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Returns the Hartree-Fock term separately for the local and non-local interaction. Since we are always SU(2)-symmetric,
    the sum over the spins of the first term in Eq. (4.55) in Anna Galler's thesis results in a simple factor of 2.
    .. math:: \Sigma_{HF}^k = 2(U_{abcd} + V^{q=0}_{abcd}) n_{dc} - 1/N_q \sum_q (U_{adcb} + V^{q}_{adcb}) n^{k-q}_{dc}
    where the Hartree-term is given by

    .. math:: \Sigma_{H} = 2(U_{abcd} + V^{q=0}_{abcd}) n_{dc}
    and the Fock-term reads

    .. math:: \Sigma_{F}^k = - 1/N_q \sum_q (U_{adcb} + V^{q}_{adcb}) n^{k-q}_{dc}.
    """
    v_q0 = v_nonloc.find_q((0, 0, 0))
    occ_qk = np.array([np.roll(config.sys.occ_k, [-i for i in q], axis=(0, 1, 2)) for q in full_q_list])  # [q,k,o1,o2]
    nq_tot, nk_tot = np.prod(config.lattice.nq), np.prod(config.lattice.nk)
    occ_qk = occ_qk.reshape(nq_tot, nk_tot, config.sys.n_bands, config.sys.n_bands)

    hartree = 2 * (u_loc + v_q0).times("qabcd,dc->ab", config.sys.occ)
    fock = (
        -1.0
        / nq_tot
        * (u_loc + v_nonloc).compress_q_dimension().permute_orbitals("abcd->adcb").times("qabcd,qkdc->kab", occ_qk)
    )
    return hartree[None, ..., None], fock[..., None]  # [k,o1,o2,v]


def get_sigma_kernel_vrg_r_q(
    vrg_r_q: FourPoint, gchi_r_q_sum: FourPoint, u_loc: LocalInteraction, v_nonloc: Interaction
) -> tuple[Interaction, FourPoint]:
    r"""
    Returns the kernel for the self-energy calculation.
    .. math:: K = -\gamma_{r;abcd}^{qv} + \gamma_{r;abef}^{qv} * U^{q}_{r;fehg} * \chi_{r;ghcd}^{q}

    Plus 2/3 times the identity if the channel is the magnetic channel, since there is an additional contribution of 2
    in the equations and the magnetic part is multiplied by 3.
    """
    u_r = v_nonloc.as_channel(vrg_r_q.channel) + u_loc.as_channel(vrg_r_q.channel)
    kernel = -vrg_r_q + vrg_r_q @ u_r @ gchi_r_q_sum
    if vrg_r_q.channel == SpinChannel.MAGN:
        kernel += 2.0 / 3.0 * FourPoint.identity_like(vrg_r_q)
    return u_r, kernel


def calculate_sde_r(u_r: Interaction, kernel_r: FourPoint, g_qk: np.ndarray) -> np.ndarray:
    r"""
    Returns
    .. math:: \Sigma_{ij}^{k} = -1/\beta \sum_q [ U^q_{r;aibc} * K_{r;cbjd}^{qv} * G_{ad}^{w-v} ],
    where

    .. math:: K_{r;abcd}^{qv} = 1 - \gamma_{r;abcd}^{qv} + \gamma_{r;abef}^{qv} * U^{q}_{r;fehg} * \chi_{r;ghcd}^{q}
    """
    return (
        0.5
        / config.sys.beta**2
        / config.lattice.q_grid.nk_tot
        * u_r.times("qrjop,qilpowv,qklrwv->kijv", kernel_r, g_qk)
    )


def get_self_energy_dc_q(
    f_dens: LocalFourPoint,
    f_magn: LocalFourPoint,
    u_loc: LocalInteraction,
    gchi0_q_core: FourPoint,
    g_qk: np.ndarray,
):
    """
    Returns the double-counting kernel, see Eq. (1.124) in my thesis.
    """

    def calc_kernel(f_r):
        return (
            (gchi0_q_core @ f_r).sum_over_vn(config.sys.beta, axis=(-2,)).map_to_full_bz(config.lattice.q_grid.irrk_inv)
        )

    kernel_dens = calc_kernel(f_dens)
    kernel_magn = calc_kernel(f_magn)

    def calc_dc(kernel_r):
        mat = (
            0.5
            / config.sys.beta**2
            / config.lattice.q_grid.nk_tot
            * u_loc.permute_orbitals("abcd->adcb").times("aibc,qcbjdwv,qkadwv->kijv", kernel_r, g_qk)
        )
        return SelfEnergy(mat, config.lattice.nk, True, True)

    return calc_dc(kernel_dens) + 3 * calc_dc(kernel_magn)


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
    )  # has [q,k,o1,o2,w,v], might be a big memory problem
    return g_qk


def gather_and_scatter(mpi_distributor: MpiDistributor, obj: FourPoint | Interaction):
    obj.mat = mpi_distributor.gather(obj.mat)
    obj.update_original_shape()
    obj.mat = mpi_distributor.scatter(obj.mat)
    return obj


def calculate_self_energy_q(
    comm: MPI.Comm,
    giwk: GreensFunction,
    gamma_dens: LocalFourPoint,
    gamma_magn: LocalFourPoint,
    f_dens: LocalFourPoint,
    f_magn: LocalFourPoint,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
) -> SelfEnergy:
    logger = config.logger
    logger.log_info("Initializing MPI distributor.")
    mpi_distributor = MpiDistributor.create_distributor(ntasks=config.lattice.q_grid.nk_irr, comm=comm, name="Q")
    comm.barrier()
    full_q_list = config.lattice.q_grid.get_q_list()
    my_irr_q_list = config.lattice.q_grid.get_irrq_list()[mpi_distributor.my_slice]

    hartree, fock = get_hartree_fock(u_loc, v_nonloc, full_q_list)
    my_v_nonloc = v_nonloc.reduce_q(my_irr_q_list)

    logger.log_info("Starting with non-local DGA routine.")
    giwk_full = giwk.get_g_full()
    del giwk

    gchi0_q = create_generalized_chi0_q(giwk_full, my_irr_q_list)  # this is for the q list of the current rank
    logger.log_info("Non-local bare susceptibility chi_0^qv done.")
    logger.log_memory_usage("gchi0_q", gchi0_q.memory_usage_in_gb, n_exists=1)

    gchi0_q_full_sum = 1.0 / config.sys.beta * gchi0_q.sum_over_all_vn(config.sys.beta)
    logger.log_info("Sum of chi_0^qv for the niv_full region done.")

    gchi0_q_core = gchi0_q.cut_niv(config.box.niv_core)
    del gchi0_q
    gchi0_q_core_inv = gchi0_q_core.invert().take_vn_diagonal()
    logger.log_info("Inverted the non-local bare susceptibility chi_0^qv in the niv_core region.")

    g_qk = get_g_qk_wv(giwk_full, full_q_list)
    sigma_dc = get_self_energy_dc_q(f_dens, f_magn, u_loc, gchi0_q_core, g_qk)
    del f_dens, f_magn, g_qk

    gchi0_q_core_sum = 1.0 / config.sys.beta * gchi0_q_core.sum_over_all_vn(config.sys.beta)
    del gchi0_q_core
    logger.log_info("Sum of chi_0^qv for the niv_core region done.")

    vrg_dens_q, gchi_dens_q_sum = create_vertex_functions_q(
        gamma_dens, gchi0_q_core_inv, gchi0_q_full_sum, gchi0_q_core_sum, u_loc, my_v_nonloc
    )
    del gamma_dens
    logger.log_info("Three-leg vertex and physical susceptibility for the density channel calculated.")

    vrg_magn_q, gchi_magn_q_sum = create_vertex_functions_q(
        gamma_magn, gchi0_q_core_inv, gchi0_q_full_sum, gchi0_q_core_sum, u_loc, my_v_nonloc
    )
    del gamma_magn, gchi0_q_core_inv, gchi0_q_full_sum, gchi0_q_core_sum, my_v_nonloc
    logger.log_info("Three-leg vertex and physical susceptibility for the magnetic channel calculated.")

    vrg_dens_q = gather_and_scatter(mpi_distributor, vrg_dens_q)
    gchi_dens_q_sum = gather_and_scatter(mpi_distributor, gchi_dens_q_sum)
    vrg_magn_q = gather_and_scatter(mpi_distributor, vrg_magn_q)
    gchi_magn_q_sum = gather_and_scatter(mpi_distributor, gchi_magn_q_sum)

    # we have to create a new distributor for the full Brillouine zone
    mpi_distributor = MpiDistributor(config.lattice.q_grid.nk_tot, comm, "FBZ")
    my_full_q_list = config.lattice.q_grid.get_q_list()[mpi_distributor.my_slice]

    vrg_dens_q = vrg_dens_q.map_to_full_bz(config.lattice.q_grid.irrk_inv).reduce_q(my_full_q_list)
    gchi_dens_q_sum = gchi_dens_q_sum.map_to_full_bz(config.lattice.q_grid.irrk_inv).reduce_q(my_full_q_list)
    v_nonloc = v_nonloc.reduce_q(my_full_q_list)

    u_dens, kernel_dens = get_sigma_kernel_vrg_r_q(vrg_dens_q, gchi_dens_q_sum, u_loc, v_nonloc)
    del vrg_dens_q, gchi_dens_q_sum
    logger.log_info("Kernel function for sigma for the density channel calculated.")

    u_dens = gather_and_scatter(mpi_distributor, u_dens)
    kernel_dens = gather_and_scatter(mpi_distributor, kernel_dens)

    g_qk = get_g_qk_wv(giwk_full, my_full_q_list)
    sigma_mat = calculate_sde_r(u_dens, kernel_dens, g_qk)
    del g_qk, u_dens, kernel_dens
    logger.log_info("First Kernel function for sigma for the density channel calculated.")

    vrg_magn_q = vrg_magn_q.map_to_full_bz(config.lattice.q_grid.irrk_inv).reduce_q(my_full_q_list)
    gchi_magn_q_sum = gchi_magn_q_sum.map_to_full_bz(config.lattice.q_grid.irrk_inv).reduce_q(my_full_q_list)

    u_magn, kernel_magn = get_sigma_kernel_vrg_r_q(vrg_magn_q, gchi_magn_q_sum, u_loc, v_nonloc)
    del vrg_magn_q, gchi_magn_q_sum
    logger.log_info("Kernel function for sigma for the magnetic channel calculated.")

    u_magn = gather_and_scatter(mpi_distributor, u_magn)
    kernel_magn = gather_and_scatter(mpi_distributor, kernel_magn)

    g_qk = get_g_qk_wv(giwk_full, my_full_q_list)
    sigma_mat += 3 * calculate_sde_r(u_magn, kernel_magn, g_qk)
    del g_qk
    del u_magn, kernel_magn
    logger.log_info("Second Kernel function for sigma for the magnetic channel calculated.")

    sigma_mat = mpi_distributor.allreduce(sigma_mat)
    sigma_mat += hartree
    sigma_mat += fock
    sigma_mat -= sigma_dc.mat

    # no dc kernel subtracted yet!
    return SelfEnergy(sigma_mat, config.lattice.nk, True, True, False)
