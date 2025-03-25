import mpi4py.MPI as MPI

import config
from four_point import FourPoint
from greens_function import GreensFunction, update_mu
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
        gchi0_q, SpinChannel.NONE, config.lattice.nq, 1, 1, full_niw_range=False, has_compressed_q_dimension=True
    )


def get_qk_single_q(giwk: GreensFunction, q: tuple) -> np.ndarray:
    """
    Returns G_{ab}^{q-k} for a single q point, shape is [k,o1,o2,w,v], where only half the niv range is returned.
    """
    shifted_mat = giwk.shift_k_by_q([-i for i in q]).mat
    return MFHelper.wn_slices_gen(shifted_mat, config.box.niv_core, config.box.niw_core).reshape(
        config.lattice.k_grid.nk_tot,
        config.sys.n_bands,
        config.sys.n_bands,
        2 * config.box.niw_core + 1,
        2 * config.box.niv_core,
    )[..., config.box.niv_core : 2 * config.box.niv_core]


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
    fock = -1.0 / nq_tot * (u_loc + v_nonloc).compress_q_dimension().times("qadcb,qkdc->kab", occ_qk)
    return hartree[None, ..., None], fock[..., None]  # [k,o1,o2,v]


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


def create_vrg_q(gchi_aux_q_r: FourPoint, gchi0_q_inv: FourPoint) -> FourPoint:
    r"""
    Returns the three-leg vertex
    .. math:: \gamma_{r;lmm'l'}^{qv} = \beta * (\chi^{qvv}_{0;lmab})^{-1} * (\sum_{v'} \chi^{*;qvv'}_{r;bam'l'}).
    See Eq. (3.71) in Paul Worm's thesis.
    """
    gchi_aux_q_r_sum = gchi_aux_q_r.sum_over_vn(config.sys.beta, axis=(-1,))
    return config.sys.beta * (gchi0_q_inv @ gchi_aux_q_r_sum).take_vn_diagonal()


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


def calculate_sigma_dc_kernel(f_dens: LocalFourPoint, f_magn: LocalFourPoint, gchi0_q_core: FourPoint) -> FourPoint:
    """
    Returns the double-counting kernel for the self-energy calculation for only half the niv range to save computation time.
    """
    return (gchi0_q_core @ (f_dens + 3 * f_magn)).sum_over_vn(config.sys.beta, axis=(-2,)).to_half_niv_range()


def calculate_kernel_r_q_from_vrg_r_q(vrg_q_r, gchi_aux_q_r_sum, v_nonloc, u_loc):
    r"""
    Returns the kernel for the self-energy calculation.
    .. math:: K = \gamma_{r;abcd}^{qv} - \gamma_{r;abef}^{qv} * U^{q}_{r;fehg} * \chi_{r;ghcd}^{q}

    minus 2/3 times the identity if the channel is the magnetic channel (due to the extra u in Eq. (1.125)).
    """
    u_r = v_nonloc.as_channel(vrg_q_r.channel) + u_loc.as_channel(vrg_q_r.channel)
    kernel = vrg_q_r - vrg_q_r @ u_r @ gchi_aux_q_r_sum
    if vrg_q_r.channel == SpinChannel.MAGN:
        kernel -= 2.0 / 3.0
    return u_r, kernel.to_half_niv_range()


def calculate_sigma_kernel_r_q(
    gamma_r: LocalFourPoint,
    gchi0_q_core_inv: FourPoint,
    gchi0_q_full_sum: FourPoint,
    gchi0_q_core_sum: FourPoint,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
) -> tuple[Interaction, FourPoint]:
    logger = config.logger

    gchi_aux_q_r = create_auxiliary_chi_q(gamma_r, gchi0_q_core_inv, u_loc, v_nonloc)
    logger.log_info(f"Non-Local auxiliary susceptibility ({gchi_aux_q_r.channel.value}) calculated.")

    vrg_q_r = create_vrg_q(gchi_aux_q_r, gchi0_q_core_inv)
    logger.log_info(f"Non-local three-leg vertex gamma^wv ({vrg_q_r.channel.value}) done.")

    gchi_aux_q_r_sum = gchi_aux_q_r.sum_over_all_vn(config.sys.beta)
    del gchi_aux_q_r

    gchi_aux_q_r_sum = create_generalized_chi_q_with_shell_correction(
        gchi_aux_q_r_sum, gchi0_q_full_sum, gchi0_q_core_sum, u_loc, v_nonloc
    )
    logger.log_info(
        f"Updated non-local susceptibility chi^q ({gchi_aux_q_r_sum.channel.value}) with asymptotic correction."
    )

    u_r, kernel = calculate_kernel_r_q_from_vrg_r_q(vrg_q_r, gchi_aux_q_r_sum, v_nonloc, u_loc)
    logger.log_info(f"Kernel ({kernel.channel.value}) for sigma calculated.")
    return u_r, kernel


def calculate_u_kernel_g(
    u_r: LocalInteraction, kernel_r: FourPoint, giwk: GreensFunction, q_list: np.ndarray
) -> SelfEnergy:
    r"""
    Returns
    .. math:: \Sigma_{ij}^{k} = -1/2 * 1/\beta * 1/N_q \sum_q [ U^q_{r;aibc} * K_{r;cbjd}^{qv} * G_{ad}^{w-v} ].
    """
    mat = np.zeros(
        (
            np.prod(config.lattice.nk),
            config.sys.n_bands,
            config.sys.n_bands,
            config.box.niv_core,
        ),
        dtype=kernel_r.mat.dtype,
    )
    for idx, q in enumerate(q_list):
        mat += np.einsum(
            "aibc,cbjdwv,kadwv->kijv",
            u_r.mat if not isinstance(u_r, Interaction) else u_r.mat[idx],
            kernel_r.mat[idx],
            get_qk_single_q(giwk, q),
            optimize=True,
        )
    prefactor = -0.5 / config.sys.beta**2 / config.lattice.q_grid.nk_tot
    mat *= prefactor
    return SelfEnergy(mat, config.lattice.nk, False, True)


def gather_from_irrk_map_to_fbz_scatter(obj, mpi_dist_irrk: MpiDistributor, mpi_dist_fbz, comm: MPI.Comm):
    """
    Gathers the objects from the irreducible Brillouin zone, maps it to the full Brillouin zone and then scatters
    it across all MPI cores.
    """
    obj.mat = mpi_dist_irrk.gather(obj.mat, 0)
    obj.update_original_shape()
    if comm.rank == 0:
        obj = obj.map_to_full_bz(config.lattice.q_grid.irrk_inv)
    obj.mat = mpi_dist_fbz.scatter(obj.mat, 0)
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
    sigma_dmft: SelfEnergy,
    sigma_local: SelfEnergy,
) -> SelfEnergy:
    logger = config.logger
    logger.log_info("Starting with non-local DGA routine.")
    logger.log_info("Initializing MPI distributor.")

    # MPI distributor for the irreducible BZ
    mpi_dist_irrk = MpiDistributor.create_distributor(ntasks=config.lattice.q_grid.nk_irr, comm=comm, name="Q")
    full_q_list = config.lattice.q_grid.get_q_list()
    my_irr_q_list = config.lattice.q_grid.get_irrq_list()[mpi_dist_irrk.my_slice]

    # MPI distributor for the full BZ
    mpi_dist_fbz = MpiDistributor(config.lattice.q_grid.nk_tot, comm, "FBZ")
    my_full_q_list = config.lattice.q_grid.get_q_list()[mpi_dist_fbz.my_slice]

    # Hartree- and Fock-terms
    v_nonloc = v_nonloc.compress_q_dimension()
    hartree, fock = get_hartree_fock(u_loc, v_nonloc, full_q_list)
    logger.log_info("Calculated Hartree and Fock terms.")
    v_nonloc = v_nonloc.reduce_q(my_irr_q_list)

    giwk_full = giwk.get_g_full_from_gloc()
    del giwk

    old_sigma = sigma_dmft

    for i in range(config.self_consistency.max_iter):
        logger.log_info("----------------------------------------")
        logger.log_info(f"Starting iteration {i + 1}.")
        logger.log_info("----------------------------------------")

        gchi0_q = create_generalized_chi0_q(giwk_full, my_irr_q_list)
        gchi0_q_full_sum = 1.0 / config.sys.beta * gchi0_q.sum_over_all_vn(config.sys.beta)
        gchi0_q_core = gchi0_q.cut_niv(config.box.niv_core)
        del gchi0_q

        gchi0_q_core_inv = gchi0_q_core.invert().take_vn_diagonal()
        gchi0_q_core_sum = 1.0 / config.sys.beta * gchi0_q_core.sum_over_all_vn(config.sys.beta)

        u_dens, kernel_dens = calculate_sigma_kernel_r_q(
            gamma_dens, gchi0_q_core_inv, gchi0_q_full_sum, gchi0_q_core_sum, u_loc, v_nonloc
        )
        u_magn, kernel_magn = calculate_sigma_kernel_r_q(
            gamma_magn, gchi0_q_core_inv, gchi0_q_full_sum, gchi0_q_core_sum, u_loc, v_nonloc
        )
        del gchi0_q_core_inv, gchi0_q_full_sum, gchi0_q_core_sum
        logger.log_info("Calculated channel-specific kernels for sigma.")

        kernel_dc = calculate_sigma_dc_kernel(f_dens, f_magn, gchi0_q_core)
        del gchi0_q_core
        logger.log_info("Double-counting kernel calculated.")

        u_dc = u_loc.permute_orbitals("abcd->adcb")
        kernel_dc = gather_from_irrk_map_to_fbz_scatter(kernel_dc, mpi_dist_irrk, mpi_dist_fbz, comm)
        sigma_dc = calculate_u_kernel_g(u_dc, kernel_dc, giwk_full, my_full_q_list)
        del kernel_dc, u_dc
        logger.log_info("Double-counting correction to sigma^k calculated.")

        u_dens = gather_from_irrk_map_to_fbz_scatter(u_dens, mpi_dist_irrk, mpi_dist_fbz, comm)
        kernel_dens = gather_from_irrk_map_to_fbz_scatter(kernel_dens, mpi_dist_irrk, mpi_dist_fbz, comm)
        new_sigma = calculate_u_kernel_g(u_dens, kernel_dens, giwk_full, my_full_q_list)
        del u_dens, kernel_dens
        logger.log_info("Sigma for the density channel calculated.")

        u_magn = gather_from_irrk_map_to_fbz_scatter(u_magn, mpi_dist_irrk, mpi_dist_fbz, comm)
        kernel_magn = gather_from_irrk_map_to_fbz_scatter(kernel_magn, mpi_dist_irrk, mpi_dist_fbz, comm)
        new_sigma += 3 * calculate_u_kernel_g(u_magn, kernel_magn, giwk_full, my_full_q_list)
        del u_magn, kernel_magn
        logger.log_info("Sigma for the magnetic channel calculated.")

        new_sigma.mat = mpi_dist_fbz.allreduce(new_sigma.mat)
        sigma_dc.mat = mpi_dist_fbz.allreduce(sigma_dc.mat)

        new_sigma = new_sigma + hartree + fock + sigma_dc
        logger.log_info("Full non-local self-energy calculated.")

        old_mu = config.sys.mu
        config.sys.mu = update_mu(
            config.sys.mu, config.sys.n, giwk_full.ek, new_sigma.mat, config.sys.beta, new_sigma.fit_smom()[0]
        )
        logger.log_info(f"Updated mu from {old_mu} to {config.sys.mu}.")

        if i == 0:
            new_sigma = new_sigma + sigma_dmft.cut_niv(config.box.niv_core) - sigma_local.cut_niv(config.box.niv_core)
        new_sigma = new_sigma.pad_with_dmft_self_energy(sigma_dmft)

        if config.self_consistency.use_poly_fit and config.poly_fitting.do_poly_fitting:
            new_sigma = new_sigma.fit_polynomial(
                config.poly_fitting.n_fit, config.poly_fitting.o_fit, config.box.niv_core
            )

        new_sigma = config.self_consistency.mixing * new_sigma + (1 - config.self_consistency.mixing) * old_sigma

        if config.self_consistency.save_iter and config.output.save_quantities and comm.rank == 0:
            new_sigma.save(name=f"sigma_dga_{i+1}", output_dir=config.output.output_path)

        giwk_full = GreensFunction.get_g_full(new_sigma, config.sys.mu, giwk_full.ek)

        if np.allclose(old_sigma.mat, new_sigma.mat, atol=config.self_consistency.epsilon):
            logger.log_info(f"Self-consistency reached. Sigma converged at iteration {i+1}.")

        old_sigma = new_sigma

    mpi_dist_irrk.delete_file()
    mpi_dist_fbz.delete_file()

    if config.poly_fitting.do_poly_fitting and not config.self_consistency.use_poly_fit:
        old_sigma = old_sigma.fit_polynomial(config.poly_fitting.n_fit, config.poly_fitting.o_fit, config.box.niv_core)
        logger.log_info(f"Fitted polynomial of degree {config.poly_fitting.o_fit} to sigma.")

    if config.output.save_quantities and config.poly_fitting.do_poly_fitting and comm.rank == 0:
        old_sigma.save(name=f"sigma_dga_fitted", output_dir=config.output.output_path)
        logger.log_info("Saved fitted sigma as numpy file.")

    return old_sigma
