import glob
import os
import re

import mpi4py.MPI as MPI
import numpy as np

import config
from four_point import FourPoint
from greens_function import GreensFunction, update_mu
from interaction import LocalInteraction, Interaction
from local_four_point import LocalFourPoint
from matsubara_frequencies import *
from mpi_distributor import MpiDistributor
from n_point_base import SpinChannel
from self_energy import SelfEnergy


def create_generalized_chi0_q(giwk: GreensFunction, q_list: np.ndarray, irrk_factor: int) -> FourPoint:
    """
    Returns gchi0^{qk}_{lmm'l'} = -beta * G^{k}_{ll'} * G^{k-q}_{m'm}
    """
    wn = MFHelper.wn(config.box.niw_core, return_only_positive=True)
    niv_full = config.box.niv_full
    niv_full_range = np.arange(-niv_full, niv_full)

    gchi0_q = np.zeros(
        (len(q_list),) + (config.sys.n_bands,) * 4 + (len(wn), 2 * niv_full),
        dtype=giwk.mat.dtype,
    )

    g_left_mat = (
        giwk.mat[:, :, :, :, None, None, :, giwk.niv + niv_full_range[None, :]]
        * np.eye(config.sys.n_bands)[None, None, None, None, :, :, None, None, None]
    )

    g_right = giwk.transpose_orbitals()
    # we do this to save memory and to avoid a full loop over all wn independently
    # since we have to have the full bz for this, we do it in batches
    # the batch size is determined by the largest object, which is gchi_aux. The factor "irrk_factor" is the difference
    # in storage between an item in the full bz and the irr bz. This way the batch size is as large as it can be
    # to not exceed memory, i.e. such that the objects are still a tiny bit smaller than gchi_aux
    batch_size = max(1, config.box.niw_core // (2 * irrk_factor))
    for batch_start in range(0, len(wn), batch_size):
        batch_end = min(batch_start + batch_size, len(wn))
        wn_batch = wn[batch_start:batch_end]

        for idx, q in enumerate(q_list):
            g_right_mat = (
                g_right.shift_k_by_q([-i for i in q]).mat[
                    :, :, :, None, :, :, None, giwk.niv + niv_full_range[None, :] - wn_batch[:, None]
                ]
                * np.eye(config.sys.n_bands)[None, None, None, :, None, None, :, None, None]
            )
            gchi0_q[idx, ..., batch_start:batch_end, :] = np.sum(g_left_mat * g_right_mat, axis=(0, 1, 2))

    gchi0_q *= -config.sys.beta / config.lattice.q_grid.nk_tot

    return FourPoint(
        gchi0_q, SpinChannel.NONE, config.lattice.nq, 1, 1, full_niw_range=False, has_compressed_q_dimension=True
    )


def get_qk_single_q(giwk: GreensFunction, q: tuple) -> np.ndarray:
    """
    Returns G_{ab}^{q-k} for a single q point, shape is [k,o1,o2,w,v].
    """
    shifted_mat = giwk.shift_k_by_q([-i for i in q]).mat
    return MFHelper.wn_slices_gen(shifted_mat, config.box.niv_core, config.box.niw_core).reshape(
        config.lattice.k_grid.nk_tot,
        config.sys.n_bands,
        config.sys.n_bands,
        2 * config.box.niw_core + 1,
        2 * config.box.niv_core,
    )


def get_hartree_fock(
    u_loc: LocalInteraction, v_nonloc: Interaction, q_list: np.ndarray
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
    occ_qk = np.array([np.roll(config.sys.occ_k, [-i for i in q], axis=(0, 1, 2)) for q in q_list])  # [q,k,o1,o2]
    nq_tot, nk_tot = np.prod(config.lattice.nq), np.prod(config.lattice.nk)
    occ_qk = occ_qk.reshape(nq_tot, nk_tot, config.sys.n_bands, config.sys.n_bands)

    hartree = 2 * (u_loc + v_q0).times("qabcd,dc->ab", config.sys.occ)
    fock = -1.0 / nq_tot * (u_loc + v_nonloc).compress_q_dimension().times("qadcb,qkdc->kab", occ_qk)
    return hartree[None, ..., None], fock[..., None]  # [k,o1,o2,v]


def create_auxiliary_chi_r_q(
    gamma_r: LocalFourPoint, gchi0_q_inv: FourPoint, u_loc: LocalInteraction, v_nonloc: Interaction
) -> FourPoint:
    r"""
    Returns the auxiliary susceptibility
    .. math:: \chi^{*;qvv'}_{r;lmm'l'} = ((\chi_{0;lmm'l'}^{qv})^{-1} + (\Gamma_{r;lmm'l'}^{wvv'}-U_{r;lmm'l'}-V_{r;lmm'l'}^q)/\beta^2)^{-1}

    See Eq. (3.68) in Paul Worm's thesis.
    """
    return (
        (gchi0_q_inv + 1.0 / config.sys.beta**2 * gamma_r)
        - 1.0 / config.sys.beta**2 * (v_nonloc.as_channel(gamma_r.channel) + u_loc.as_channel(gamma_r.channel))
    ).invert()


def create_auxiliary_chi_r_q_sum(
    gamma_r: LocalFourPoint,
    gchi0_q: FourPoint,
    gchi0_q_inv: FourPoint,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
    summands: int = -1,
) -> FourPoint:
    if summands <= 0:
        return create_auxiliary_chi_r_q(gamma_r, gchi0_q_inv, u_loc, v_nonloc)

    factor = gamma_r - v_nonloc.as_channel(gamma_r.channel) - u_loc.as_channel(gamma_r.channel)
    factor *= 1.0 / config.sys.beta**2
    factor = (
        FourPoint(factor, gamma_r.channel, config.lattice.nq, 1, 2, False, False, v_nonloc.has_compressed_q_dimension)
        @ gchi0_q
    )
    bse_sum = gchi0_q
    next_value = gchi0_q
    for _ in range(summands - 1):
        next_value = -factor @ next_value
        bse_sum += next_value
    return bse_sum


def create_vrg_r_q(gchi_aux_q_r: FourPoint, gchi0_q_inv: FourPoint) -> FourPoint:
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


def calculate_sigma_dc_kernel(
    f_1dens_3magn: LocalFourPoint, gchi0_q_core: FourPoint, u_loc: LocalInteraction
) -> FourPoint:
    """
    Returns the double-counting kernel for the self-energy calculation for only half the niv range to save computation time.
    """
    return (
        1.0
        / config.sys.beta
        * u_loc.permute_orbitals("abcd->adcb")
        @ (gchi0_q_core @ f_1dens_3magn).sum_over_vn(config.sys.beta, axis=(-2,)).cut_niv(config.box.niv_core)
    )


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
    return u_r @ kernel


def calculate_sigma_kernel_r_q(
    gamma_r: LocalFourPoint,
    gchi0_q_core_inv: FourPoint,
    gchi0_q_full_sum: FourPoint,
    gchi0_q_core_sum: FourPoint,
    u_loc: LocalInteraction,
    v_nonloc: Interaction,
) -> FourPoint:
    logger = config.logger

    gchi_aux_q_r = create_auxiliary_chi_r_q(gamma_r, gchi0_q_core_inv, u_loc, v_nonloc)
    logger.log_info(f"Non-Local auxiliary susceptibility ({gchi_aux_q_r.channel.value}) calculated.")
    logger.log_memory_usage(f"Gchi_aux ({gchi_aux_q_r.channel.value})", gchi_aux_q_r.memory_usage_in_gb, 1)

    vrg_q_r = create_vrg_r_q(gchi_aux_q_r, gchi0_q_core_inv)
    logger.log_info(f"Non-local three-leg vertex gamma^wv ({vrg_q_r.channel.value}) done.")
    logger.log_memory_usage(f"Three-leg vertex ({vrg_q_r.channel.value})", vrg_q_r.memory_usage_in_gb, 1)

    gchi_aux_q_r_sum = gchi_aux_q_r.sum_over_all_vn(config.sys.beta)
    del gchi_aux_q_r

    gchi_aux_q_r_sum = create_generalized_chi_q_with_shell_correction(
        gchi_aux_q_r_sum, gchi0_q_full_sum, gchi0_q_core_sum, u_loc, v_nonloc
    )
    logger.log_info(
        f"Updated non-local susceptibility chi^q ({gchi_aux_q_r_sum.channel.value}) with asymptotic correction."
    )

    return calculate_kernel_r_q_from_vrg_r_q(vrg_q_r, gchi_aux_q_r_sum, v_nonloc, u_loc)


def calculate_sigma_from_kernel(kernel_r: FourPoint, giwk: GreensFunction, q_list: np.ndarray) -> SelfEnergy:
    r"""
    Returns
    .. math:: \Sigma_{ij}^{k} = -1/2 * 1/\beta * 1/N_q \sum_q [ U^q_{r;aibc} * K_{r;cbjd}^{qv} * G_{ad}^{w-v} ].
    """
    mat = np.zeros(
        (
            np.prod(config.lattice.nk),
            config.sys.n_bands,
            config.sys.n_bands,
            2 * config.box.niv_core,
        ),
        dtype=kernel_r.mat.dtype,
    )
    kernel_r = kernel_r.to_full_niw_range()

    for idx, q in enumerate(q_list):
        g = get_qk_single_q(giwk, q)
        mat += np.einsum("aijdwv,kadwv->kijv", kernel_r.mat[idx], g, optimize=True)
    prefactor = -0.5 / config.sys.beta / config.lattice.q_grid.nk_tot
    mat *= prefactor
    return SelfEnergy(mat, config.lattice.nk, True, True)


def get_starting_sigma(output_path: str, default_sigma: SelfEnergy) -> tuple[SelfEnergy, int]:
    """
    If the output directory is specified to be the same directory as was used by a previous calculation, we try to
    retrieve the last calculated self-energy as a starting point for the next calculation. If no sigma_dga_N.npy file
    is found, we return the dmft self-energy as a starting point.
    """
    if output_path == "" or output_path is None or not os.path.exists(output_path):
        return default_sigma, 0

    files = glob.glob(os.path.join(output_path, "sigma_dga_iteration_*.npy"))
    if not files:
        return default_sigma, 0

    iterations = [int(match.group(1)) for f in files if (match := re.search(r"sigma_dga_iteration_(\d+)\.npy$", f))]
    if not iterations:
        return default_sigma, 0

    max_iter = max(iterations)
    mat = np.load(os.path.join(output_path, f"sigma_dga_iteration_{max_iter}.npy"))
    return SelfEnergy(mat, config.lattice.nk, True, True, False), max_iter


def calculate_self_energy_q(
    comm: MPI.Comm,
    giwk: GreensFunction,
    gamma_dens: LocalFourPoint,
    gamma_magn: LocalFourPoint,
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
    irrk_q_list = config.lattice.q_grid.get_irrq_list()
    my_irr_q_list = irrk_q_list[mpi_dist_irrk.my_slice]
    irrk_factor = len(full_q_list) // len(irrk_q_list)

    mpi_dist_fullbz = MpiDistributor.create_distributor(ntasks=config.lattice.q_grid.nk_tot, comm=comm, name="FBZ")
    my_full_q_list = full_q_list[mpi_dist_fullbz.my_slice]

    # Hartree- and Fock-terms
    v_nonloc = v_nonloc.compress_q_dimension()
    if comm.rank == 0:
        hartree, fock = get_hartree_fock(u_loc, v_nonloc, full_q_list)
    else:
        hartree, fock = None, None
    hartree, fock = comm.bcast((hartree, fock), root=0)
    logger.log_info("Calculated Hartree and Fock terms.")
    v_nonloc = v_nonloc.reduce_q(my_irr_q_list)

    sigma_old, starting_iter = get_starting_sigma(config.self_consistency.previous_sc_path, sigma_dmft)
    if starting_iter > 0:
        logger.log_info(
            f"Using previous calculation and starting the self-consistency loop at iteration {starting_iter+1}."
        )

    for i in range(starting_iter, starting_iter + config.self_consistency.max_iter):
        logger.log_info("----------------------------------------")
        logger.log_info(f"Starting iteration {i + 1}.")
        logger.log_info("----------------------------------------")

        giwk_full = GreensFunction.get_g_full(sigma_old, config.sys.mu, giwk.ek)
        logger.log_memory_usage("giwk", giwk_full.memory_usage_in_gb, 1)
        gchi0_q = create_generalized_chi0_q(giwk_full, my_irr_q_list, irrk_factor)
        logger.log_memory_usage("Gchi0_q_full", gchi0_q.memory_usage_in_gb, 1)

        f_1dens_3magn = LocalFourPoint.load(os.path.join(config.output.output_path, "f_1dens_3magn.npy"))
        kernel = -calculate_sigma_dc_kernel(f_1dens_3magn, gchi0_q, u_loc)
        del f_1dens_3magn
        logger.log_info("Calculated double-counting kernel.")

        gchi0_q_full_sum = 1.0 / config.sys.beta * gchi0_q.sum_over_all_vn(config.sys.beta)
        gchi0_q_core = gchi0_q.cut_niv(config.box.niv_core)
        del gchi0_q
        logger.log_memory_usage("Gchi0_q_core", gchi0_q_core.memory_usage_in_gb, 1)

        gchi0_q_core_inv = gchi0_q_core.invert().take_vn_diagonal()
        logger.log_memory_usage("Gchi0_q_inv", gchi0_q_core_inv.memory_usage_in_gb, 1)
        gchi0_q_core_sum = 1.0 / config.sys.beta * gchi0_q_core.sum_over_all_vn(config.sys.beta)
        del gchi0_q_core

        kernel += calculate_sigma_kernel_r_q(
            gamma_dens, gchi0_q_core_inv, gchi0_q_full_sum, gchi0_q_core_sum, u_loc, v_nonloc
        )
        logger.log_info("Calculated kernel for density channel.")

        kernel += 3 * calculate_sigma_kernel_r_q(
            gamma_magn, gchi0_q_core_inv, gchi0_q_full_sum, gchi0_q_core_sum, u_loc, v_nonloc
        )
        del gchi0_q_core_inv, gchi0_q_full_sum, gchi0_q_core_sum
        logger.log_info("Calculated kernel for magnetic channel.")

        kernel.mat = mpi_dist_irrk.gather(kernel.mat)
        kernel = kernel.map_to_full_bz(config.lattice.q_grid.irrk_inv)
        kernel.mat = mpi_dist_fullbz.scatter(kernel.mat)
        sigma_new = calculate_sigma_from_kernel(kernel, giwk_full, my_full_q_list)
        logger.log_info("Self-energy calculated from kernel.")

        sigma_new.mat = mpi_dist_irrk.allreduce(sigma_new.mat)
        logger.log_memory_usage("Non-local sigma", sigma_new.memory_usage_in_gb, 1)

        sigma_new = sigma_new + hartree + fock
        logger.log_info("Full non-local self-energy calculated.")

        old_mu = config.sys.mu
        if comm.rank == 0:
            config.sys.mu = update_mu(
                config.sys.mu, config.sys.n, giwk_full.ek, sigma_new.mat, config.sys.beta, sigma_new.fit_smom()[0]
            )
        config.sys.mu = comm.bcast(config.sys.mu)
        logger.log_info(f"Updated mu from {old_mu} to {config.sys.mu}.")

        if i == 0:
            sigma_new = sigma_new + sigma_dmft.cut_niv(config.box.niv_core) - sigma_local.cut_niv(config.box.niv_core)
        sigma_new = sigma_new.pad_with_dmft_self_energy(sigma_dmft)

        if config.self_consistency.use_poly_fit and config.poly_fitting.do_poly_fitting:
            sigma_new = sigma_new.fit_polynomial(
                config.poly_fitting.n_fit, config.poly_fitting.o_fit, config.box.niv_core
            )
            logger.log_info(f"Fitted polynomial to sigma at iteration {i+1}.")

        sigma_new = config.self_consistency.mixing * sigma_new + (1 - config.self_consistency.mixing) * sigma_old
        logger.log_info(
            f"Sigma mixed with previous iteration using a mixing parameter of {config.self_consistency.mixing}."
        )

        if config.self_consistency.save_iter and config.output.save_quantities and comm.rank == 0:
            sigma_new.save(name=f"sigma_dga_iteration_{i+1}", output_dir=config.output.output_path)
            logger.log_info(f"Saved sigma for iteration {i+1} as numpy array.")

        logger.log_info("Checking self-consistency convergence.")
        if comm.rank == 0:
            converged = np.allclose(sigma_old.mat[0], sigma_new.mat[0], atol=config.self_consistency.epsilon)
        else:
            converged = False
        converged = comm.bcast(converged)

        sigma_old = sigma_new
        if converged:
            logger.log_info(f"Self-consistency reached. Sigma converged at iteration {i+1}.")
            break
        logger.log_info("Self-consistency not reached yet.")

    mpi_dist_irrk.delete_file()
    return sigma_old
