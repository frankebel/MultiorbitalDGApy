import config
from greens_function import GreensFunction
from interaction import LocalInteraction
from local_four_point import LocalFourPoint
from matsubara_frequencies import MFHelper, FrequencyShift
from n_point_base import *
from self_energy import SelfEnergy


def create_generalized_chi(g2: LocalFourPoint, g_loc: GreensFunction) -> LocalFourPoint:
    r"""
    Returns the generalized susceptibility
    .. math:: \chi_{r;lmm'l'}^{wvv'} = \beta (G_{r;lmm'l'}^{(2);wvv'} - 2 \delta_{rd} \delta_{w0} G_{ll'}^{v} G_{m'm}^{v'})
    """
    chi = config.sys.beta * g2

    if g2.channel == SpinChannel.DENS and g2.frequency_notation == FrequencyNotation.PH:
        wn = MFHelper.wn(config.box.niw_core)
        ggv_mat = _get_ggv_mat(g_loc, niv_slice=config.box.niv_core)[:, :, :, :, None, ...]
        chi[:, :, :, :, wn == 0, ...] -= 2.0 * config.sys.beta * ggv_mat

    return chi


def _get_ggv_mat(g_loc: GreensFunction, niv_slice: int = -1) -> np.ndarray:
    r"""
    Returns the product of two Green's functions
    .. math:: B_{0;lmm'l'}^{v} = G_{ll'}^{v} G_{m'm}^{v}.
    """
    if niv_slice == -1:
        niv_slice = g_loc.niv
    g_loc_slice_mat = g_loc.mat[..., g_loc.niv - niv_slice : g_loc.niv + niv_slice]
    g_left_mat = g_loc_slice_mat[:, None, None, :, :, None] * np.eye(g_loc.n_bands)[None, :, :, None, None, None]
    g_right_mat = (
        np.swapaxes(g_loc_slice_mat, 0, 1)[None, :, :, None, None, :]
        * np.eye(g_loc.n_bands)[:, None, None, :, None, None]
    )
    return g_left_mat * g_right_mat


def create_generalized_chi0(
    g_loc: GreensFunction, frequency_shift: FrequencyShift = FrequencyShift.MINUS
) -> LocalFourPoint:
    r"""
    Returns the generalized bare susceptibility
    .. math:: \chi_{0;lmm'l'}^{wv} = -\beta G_{ll'}^{v} G_{m'm}^{v-w}.
    """
    wn = MFHelper.wn(config.box.niw_core)
    iws, iws2 = np.array([MFHelper.get_frequency_shift(wn_i, frequency_shift) for wn_i in wn], dtype=int).T

    niv_range = np.arange(-config.box.niv_full, config.box.niv_full)
    g_left_mat = (
        g_loc.mat[:, None, None, :, g_loc.niv + niv_range[None, :] + iws[:, None]]
        * np.eye(g_loc.n_bands)[None, :, :, None, None, None]
    )
    g_right_mat = (
        g_loc.mat[None, :, :, None, g_loc.niv + niv_range[None, :] + iws2[:, None]]
        * np.eye(g_loc.n_bands)[:, None, None, :, None, None]
    )
    return LocalFourPoint(-config.sys.beta * g_left_mat * g_right_mat, SpinChannel.NONE, 1, 1)


def create_gamma_0(u_loc: LocalInteraction) -> LocalInteraction:
    r"""
    Returns the zero-th order vertex
    .. math:: \Gamma_{0;lmm'l'} = U_{lmm'l'} - U_{ll'm'm}.
    """
    return u_loc - u_loc.permute_orbitals("abcd->adcb")


def create_gamma_r(gchi_r: LocalFourPoint, gchi0: LocalFourPoint) -> LocalFourPoint:
    r"""
    Returns the irreducible vertex
    .. math:: \Gamma_{r;lmm'l'}^{wvv'} = \beta^2 * [(\chi_{r;lmm'l'}^{wvv'})^{-1} - (\chi_{0;lmm'l'}^{wv})^{-1}]
    """
    return config.sys.beta**2 * (~gchi_r - ~(gchi0.cut_niv(config.box.niv_core)))


def create_gamma_r_with_shell_correction(
    gchi_r: LocalFourPoint, gchi0: LocalFourPoint, u_loc: LocalInteraction
) -> LocalFourPoint:
    """
    Calculates the irreducible vertex with the shell correction as described
    by Motoharu Kitatani et al 2022 J. Phys. Mater. 5 034005; DOI 10.1088/2515-7639/ac7e6d.
    More specifically equations A.4 to A.8
    """
    chi_tilde_shell = ~(~gchi0 + 1.0 / config.sys.beta**2 * u_loc.as_channel(gchi_r.channel))
    chi_tilde_core_inv = ~(chi_tilde_shell.cut_niv(config.box.niv_core))
    return config.sys.beta**2 * (~gchi_r - chi_tilde_core_inv) + u_loc.as_channel(gchi_r.channel)


def create_auxiliary_chi(gamma_r: LocalFourPoint, gchi_0: LocalFourPoint, u_loc: LocalInteraction) -> LocalFourPoint:
    r"""
    Returns the auxiliary susceptibility
    .. math:: \chi^{*;wvv'}_{r;lmm'l'} = (\chi_{0;lmm'l'}^{-1} + (\Gamma_{r;lmm'l'}-\Gamma_{0;lmm'l'})/\beta^2)^{-1}.
    See Eq. (3.68) in Paul Worm's thesis.
    """
    return ~(~gchi_0.cut_niv(config.box.niv_core) + (gamma_r - u_loc.as_channel(gamma_r.channel)) / config.sys.beta**2)


def create_physical_chi(gchi_r: LocalFourPoint) -> LocalFourPoint:
    r"""
    Returns the physical susceptibility
    .. math:: \chi_{r;ll'}^{phys;w} = \sum_{vv'} \sum_{mm'} \chi_{r;lmm'l'}^{wvv'}.
    See Eq. (3.51) in Paul Worm's thesis.
    """
    return gchi_r.contract_legs(config.sys.beta)


def create_vrg(gchi_aux: LocalFourPoint, gchi0: LocalFourPoint) -> LocalFourPoint:
    r"""
    Returns the three-leg vertex
    .. math:: \gamma_{r;lmm'l'} = \beta * (\chi^{wvv}_{0;lmab})^{-1} * (\sum_{v'} \chi^{*;wvv'}_{r;bam'l'}).
    See Eq. (3.71) in Paul Worm's thesis.
    """
    gchi_aux_sum = gchi_aux.sum_over_vn(config.sys.beta, axis=(-1,))
    return config.sys.beta * (~gchi0.cut_niv(config.box.niv_core) @ gchi_aux_sum).compress_vn_dimensions()


def get_loc_self_energy_vrg(
    vrg_dens: LocalFourPoint,
    vrg_magn: LocalFourPoint,
    gchi_dens: LocalFourPoint,
    gchi_magn: LocalFourPoint,
    g_loc: GreensFunction,
    u_loc: LocalInteraction,
) -> SelfEnergy:
    """
    Performs the local self-energy calculation using the Schwinger-Dyson equation,
    see Paul Worm's thesis, Eq. (3.70) and Anna Galler's Thesis, P. 76 ff.
    """
    # 1=i, 2=j, 3=k, 4=l, 7=o, 8=p
    g_1 = MFHelper.wn_slices_gen(g_loc.mat, config.box.niv_core, config.box.niw_core)
    inner_dens = vrg_dens - (vrg_dens @ u_loc.as_channel(SpinChannel.DENS) @ gchi_dens)
    inner_magn = vrg_magn - (vrg_magn @ u_loc.as_channel(SpinChannel.MAGN) @ gchi_magn)
    self_energy_mat = (
        -1.0 / config.sys.beta * u_loc.times("kjop,ilpowv,lkwv->ijv", 0.5 * (inner_dens - inner_magn), g_1)
    )
    hartree = u_loc.as_channel(SpinChannel.DENS).times("abcd,dc->ab", config.sys.occ)[..., None]

    return SelfEnergy(self_energy_mat + hartree)


def get_loc_self_energy_gamma_abinitio_dga(
    gamma_dens: LocalFourPoint, u_loc: LocalInteraction, g_loc: GreensFunction
) -> SelfEnergy:
    r"""
    Returns the local self-energy with the three-leg gamma from AbinitioDGA
    .. math:: \Sigma_{ij}^{v} = -1/\beta \sum_w [ U_{iabc} * \gamma_{cbdj}^{wv} * G_{ad}^{w-v} ]
    """
    g_1 = MFHelper.wn_slices_gen(g_loc.mat, config.box.niv_core, config.box.niw_core)
    sigma = -1.0 / config.sys.beta * u_loc.times("iabc,cbdjwv,adwv->ijv", gamma_dens, g_1)
    hartree = u_loc.as_channel(SpinChannel.DENS).times("abcd,dc->ab", config.sys.occ)[..., None]
    return SelfEnergy(sigma + hartree)


def create_generalized_chi_with_asympt_correction(
    gchi_aux_sum: LocalFourPoint, gchi0: LocalFourPoint, u_loc: LocalInteraction
) -> LocalFourPoint:
    gchi0_full_sum = gchi0.sum_over_all_vn(config.sys.beta)
    gchi0_core_sum = gchi0.cut_niv(config.box.niv_core).sum_over_all_vn(config.sys.beta)
    test = gchi_aux_sum + gchi0_full_sum - gchi0_core_sum
    return ~(~(gchi_aux_sum + gchi0_full_sum - gchi0_core_sum) + u_loc / config.sys.beta**2)


def perform_local_schwinger_dyson(
    g_loc: GreensFunction, g2_dens: LocalFourPoint, g2_magn: LocalFourPoint, u_loc: LocalInteraction
):
    """
    ATTENTION: THIS IS PAUL'S ROUTINE
    Performs the local Schwinger-Dyson equation calculation for the local (DMFT) self-energy for sanity checks.
    Includes the calculation of the three-leg vertices, (auxiliary/bare/physical) susceptibilities and the irreducible vertices.
    """
    logger = config.logger

    gchi_dens = create_generalized_chi(g2_dens, g_loc)
    del g2_dens
    logger.log_info("Local generalized susceptibility chi^wvv' (dens) done.")
    gchi_magn = create_generalized_chi(g2_magn, g_loc)
    del g2_magn
    logger.log_info("Local generalized susceptibility chi^wvv' (magn) done.")

    if config.output.do_plotting and config.current_rank == 0:
        gchi_dens.plot(omega=0, name=f"Gchi_dens", output_dir=config.output.output_path)
        gchi_magn.plot(omega=0, name=f"Gchi_magn", output_dir=config.output.output_path)
        logger.log_info("Local generalized susceptibilities plotted.")

    gchi0 = create_generalized_chi0(g_loc)

    # Density objects

    gamma_dens = create_gamma_r_with_shell_correction(gchi_dens, gchi0, u_loc)
    logger.log_info("Local irreducible vertex Gamma^wvv' (dens) with asymptotic correction done.")

    chi_dens_physical = gchi_dens.contract_legs(config.sys.beta)
    logger.log_info("Local physical susceptibility chi^w (dens) done.")
    # del gchi_dens

    gchi_dens_aux = create_auxiliary_chi(gamma_dens, gchi0, u_loc)
    logger.log_info("Local auxiliary susceptibility chi^*wvv' (dens) done.")

    vrg_dens = create_vrg(gchi_dens_aux, gchi0)
    logger.log_info("Local three-leg vertex gamma^wv (dens) done.")

    gchi_dens_aux_sum = gchi_dens_aux.sum_over_all_vn(config.sys.beta)
    # gchi_dens_sum = gchi_dens.sum_over_all_vn(config.sys.beta)
    gchi_dens_sum = create_generalized_chi_with_asympt_correction(gchi_dens_aux_sum, gchi0, u_loc)
    logger.log_info("Updated local generalized susceptibility chi^wvv' (dens) with asymptotic correction.")
    del gchi_dens_aux

    # Magnetic objects

    gamma_magn = create_gamma_r_with_shell_correction(gchi_magn, gchi0, u_loc)
    logger.log_info("Local irreducible vertex Gamma^wvv' (magn) with asymptotic correction done.")
    chi_magn_physical = gchi_magn.contract_legs(config.sys.beta)
    logger.log_info("Local physical susceptibility chi^w (magn) done.")
    # del gchi_magn

    gchi_magn_aux = create_auxiliary_chi(gamma_magn, gchi0, u_loc)
    logger.log_info("Local auxiliary susceptibility chi^*wvv' (magn) done.")

    vrg_magn = create_vrg(gchi_magn_aux, gchi0)
    logger.log_info("Local three-leg vertex gamma^wv (magn) done.")

    gchi_magn_aux_sum = gchi_magn_aux.sum_over_all_vn(config.sys.beta)
    # gchi_magn_sum = gchi_magn.sum_over_all_vn(config.sys.beta)
    gchi_magn_sum = create_generalized_chi_with_asympt_correction(gchi_magn_aux_sum, gchi0, u_loc)
    logger.log_info("Updated local generalized susceptibility chi^wvv' (dens) with asymptotic correction.")
    del gchi_magn_aux, gchi0

    # Sigma

    sigma = get_loc_self_energy_vrg(vrg_dens, vrg_magn, gchi_dens_sum, gchi_magn_sum, g_loc, u_loc)
    logger.log_info("Self-energy Sigma^v done.")

    return gamma_dens, gamma_magn, chi_dens_physical, chi_magn_physical, vrg_dens, vrg_magn, sigma


def create_asympt_f(
    gchi_dens: LocalFourPoint, gchi_magn: LocalFourPoint, gchi_ud_pp_sum: LocalFourPoint, u_loc: LocalInteraction
) -> LocalFourPoint:
    gchi_ud_pp_sum_from_ph = (
        (0.5 * (gchi_dens - gchi_magn)).change_frequency_notation_ph_to_pp().sum_over_all_vn(config.sys.beta)
    )
    test = gchi_dens.compress_vn_dimensions().sum_over_all_vn(config.sys.beta)
    gchi_ud_pp_sum = gchi_ud_pp_sum.cut_niw(gchi_ud_pp_sum_from_ph.niw)

    return 0


def perform_local_schwinger_dyson_abinitio_dga(
    g_loc: GreensFunction,
    g2_dens: LocalFourPoint,
    g2_magn: LocalFourPoint,
    g2_ud_pp: LocalFourPoint,
    u_loc: LocalInteraction,
):
    """
    ATTENTION: THIS IS THE ABINITODGA ROUTINE!
    Performs the local Schwinger-Dyson equation calculation for the local (DMFT) self-energy for sanity checks.
    """
    logger = config.logger

    gchi_dens_loc = create_generalized_chi(g2_dens, g_loc)
    logger.log_info("Generalized susceptibility chi^wvv' (dens) done.")
    del g2_dens
    gchi_magn_loc = create_generalized_chi(g2_magn, g_loc)
    logger.log_info("Generalized susceptibility chi^wvv' (magn) done.")
    del g2_magn
    gchi_ud_pp_loc_sum = create_generalized_chi(g2_ud_pp, g_loc).sum_over_vn(config.sys.beta, axis=(-1, -2))
    logger.log_info("Generalized susceptibility chi^wvv' (ud_pp) done.")
    del g2_ud_pp

    gchi0_loc_full = create_generalized_chi0(g_loc)
    logger.log_info("Generalized bare susceptibility chi_0^wv done.")
    gchi0_core = gchi0_loc_full.cut_niv(config.box.niv_core)

    # 1 + chi0 * F_r = gchi_r * (chi0)^(-1) = 1 + gamma_r or
    # F_r = -beta^2 * [chi0^(-1) - chi0^(-1) chi_r chi0^(-1)]
    # gamma_r is NOT the irreducible vertex in channel r but rather the three-point vertex from AbinitioDGA
    gchi0_inv_core = ~gchi0_core
    f_dens_loc = -(gchi0_inv_core - config.sys.beta**2 * gchi0_inv_core @ gchi_dens_loc @ gchi0_inv_core)
    logger.log_info("Local full vertex F^wvv' (dens) done.")
    f_magn_loc = -config.sys.beta**2 * (gchi0_inv_core - gchi0_inv_core @ gchi_magn_loc @ gchi0_inv_core)
    logger.log_info("Local full vertex F^wvv' (magn) done.")
    del gchi0_inv_core

    # f_dens_loc_with_asympt = create_asympt_f(gchi_dens_loc, gchi_magn_loc, gchi_ud_pp_loc_sum, u_loc)

    # in most equations we need 1 + gamma_r so we add it here
    one = LocalFourPoint.identity(config.sys.n_bands, config.box.niw_core, config.box.niv_full, 1)
    gamma_dens_loc = 1.0 / config.sys.beta * (gchi0_core @ f_dens_loc).sum_over_vn(config.sys.beta, axis=(-2,))
    one_plus_gamma_dens_loc = one + gamma_dens_loc
    logger.log_info("Local three-leg vertex gamma^wv (dens) done.")

    gamma_magn_loc = 1.0 / config.sys.beta * (gchi0_core @ f_magn_loc).sum_over_vn(config.sys.beta, axis=(-2,))
    one_plus_gamma_magn_loc = one + gamma_magn_loc
    logger.log_info("Local three-leg vertex gamma^wv (magn) done.")
    del gchi0_core, gamma_magn_loc

    sigma_loc = get_loc_self_energy_gamma_abinitio_dga(gamma_dens_loc, u_loc, g_loc)
    logger.log_info("Local self-energy done.")
    del gamma_dens_loc

    return (
        gchi_dens_loc,
        gchi_magn_loc,
        gchi0_loc_full,
        one_plus_gamma_dens_loc,
        one_plus_gamma_magn_loc,
        f_dens_loc,
        f_magn_loc,
        sigma_loc,
    )
