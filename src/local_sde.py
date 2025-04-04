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


def create_gamma_r(gchi_r: LocalFourPoint, gchi0_inv: LocalFourPoint) -> LocalFourPoint:
    r"""
    Returns the irreducible vertex
    .. math:: \Gamma_{r;lmm'l'}^{wvv'} = \beta^2 * [(\chi_{r;lmm'l'}^{wvv'})^{-1} - (\chi_{0;lmm'l'}^{wv})^{-1}]
    """
    return config.sys.beta**2 * (gchi_r.invert() - gchi0_inv)


def create_gamma_r_with_shell_correction(
    gchi_r: LocalFourPoint, gchi0: LocalFourPoint, u_loc: LocalInteraction
) -> LocalFourPoint:
    """
    Calculates the irreducible vertex with the shell correction as described
    by Motoharu Kitatani et al. 2022 J. Phys. Mater. 5 034005; DOI 10.1088/2515-7639/ac7e6d.
    More specifically equations A.4 to A.8. Has an additional factor of beta^2 compared to Paul Worm's code.
    """
    chi_tilde_shell = (gchi0.invert() + 1.0 / config.sys.beta**2 * u_loc.as_channel(gchi_r.channel)).invert()
    chi_tilde_core_inv = chi_tilde_shell.cut_niv(config.box.niv_core).invert()
    return config.sys.beta**2 * (gchi_r.invert() - chi_tilde_core_inv) + u_loc.as_channel(gchi_r.channel)


def create_auxiliary_chi(gamma_r: LocalFourPoint, gchi0_inv: LocalFourPoint, u_loc: LocalInteraction) -> LocalFourPoint:
    r"""
    Returns the auxiliary susceptibility
    .. math:: \chi^{*;wvv'}_{r;lmm'l'} = ((\chi_{0;lmm'l'}^{wv})^{-1} + (\Gamma_{r;lmm'l'}^{wvv'}-U_{r;lmm'l'})/\beta^2)^{-1}.
    See Eq. (3.68) in Paul Worm's thesis.
    """
    return (gchi0_inv + (gamma_r - u_loc.as_channel(gamma_r.channel)) / config.sys.beta**2).invert()


def create_generalized_chi_with_shell_correction(
    gchi_aux_sum: LocalFourPoint, gchi0: LocalFourPoint, u_loc: LocalInteraction
) -> LocalFourPoint:
    """
    Calculates the generalized susceptibility with the shell correction as described by
    Motoharu Kitatani et al. 2022 J. Phys. Mater. 5 034005; DOI 10.1088/2515-7639/ac7e6d. Eq. A.15
    """
    gchi0_full_sum = 1.0 / config.sys.beta * gchi0.sum_over_all_vn(config.sys.beta)
    gchi0_core_sum = 1.0 / config.sys.beta * gchi0.cut_niv(config.box.niv_core).sum_over_all_vn(config.sys.beta)
    return ((gchi_aux_sum + gchi0_full_sum - gchi0_core_sum).invert() + u_loc.as_channel(gchi_aux_sum.channel)).invert()


def create_full_vertex_from_gamma(gamma_r, gchi0, u_loc):
    """
    Returns the full vertex in the niv_full region. F = Gamma [1 + X_0 Gamma]^(-1)
    """
    gamma_urange = gamma_r.pad_with_u(u_loc.as_channel(gamma_r.channel), config.box.niv_full)
    return (
        gamma_urange
        @ (LocalFourPoint.identity_like(gamma_urange) + 1.0 / config.sys.beta**2 * gchi0 @ gamma_urange).invert()
    )


def create_full_vertex(gchi_r: LocalFourPoint, gchi0_inv: LocalFourPoint) -> LocalFourPoint:
    r"""
    Returns the full vertex
    .. math:: F_{r;abcd}^{wvv'} = -\beta^2 * (\chi_{0;abcd}^{-1} - \chi_{0;abef}^{-1} \chi_{r;fehg} \chi_{0;ghcd}^{-1})
    See Eq. (3.64) in Paul Worm's thesis.
    """
    return config.sys.beta**2 * (gchi0_inv - gchi0_inv @ gchi_r @ gchi0_inv)


def create_vrg(gchi_aux: LocalFourPoint, gchi0_inv: LocalFourPoint) -> LocalFourPoint:
    r"""
    Returns the three-leg vertex
    .. math:: \gamma_{r;lmm'l'}^{wv} = \beta * (\chi^{wvv}_{0;lmab})^{-1} * (\sum_{v'} \chi^{*;wvv'}_{r;bam'l'}).
    See Eq. (3.71) in Paul Worm's thesis.
    """
    gchi_aux_sum = gchi_aux.sum_over_vn(config.sys.beta, axis=(-1,))
    return config.sys.beta * (gchi0_inv @ gchi_aux_sum).take_vn_diagonal()


def create_vertex_functions(
    g2_r: LocalFourPoint,
    gchi0: LocalFourPoint,
    gchi0_inv_core: LocalFourPoint,
    g_loc: GreensFunction,
    u_loc: LocalInteraction,
) -> tuple[LocalFourPoint, LocalFourPoint, LocalFourPoint, LocalFourPoint]:
    """
    Calculates the three-leg vertex, the auxiliary susceptibility and the irreducible vertex. Employs explicit
    asymptotics as proposed by Motoharu Kitatani et al. 2022 J. Phys. Mater. 5 034005; DOI 10.1088/2515-7639/ac7e6d.
    Attention: This method will delete g2_r in the process of executing.
    """
    logger = config.logger

    gchi_r = create_generalized_chi(g2_r, g_loc)
    del g2_r
    logger.log_info(f"Local generalized susceptibility chi^wvv' ({gchi_r.channel.value}) done.")

    if config.output.do_plotting and config.current_rank == 0:
        gchi_r.plot(omega=0, name=f"Gchi_{gchi_r.channel.value}", output_dir=config.output.output_path)
        logger.log_info(f"Local generalized susceptibility ({gchi_r.channel.value}) plotted.")

    gamma_r = create_gamma_r_with_shell_correction(gchi_r, gchi0, u_loc)
    gchi0 = gchi0.take_vn_diagonal()
    logger.log_info(f"Local irreducible vertex Gamma^wvv' ({gamma_r.channel.value}) with asymptotic correction done.")

    # f_r = create_full_vertex(gchi_r, gchi0_inv_core)
    f_r = create_full_vertex_from_gamma(gamma_r, gchi0, u_loc)
    logger.log_info(f"Local full vertex F^wvv' ({f_r.channel.value}) done.")
    del gchi_r

    gchi_r_aux = create_auxiliary_chi(gamma_r, gchi0_inv_core, u_loc)
    logger.log_info(f"Local auxiliary susceptibility chi^*wvv' ({gchi_r_aux.channel.value}) done.")

    vrg_r = create_vrg(gchi_r_aux, gchi0_inv_core)
    logger.log_info(f"Local three-leg vertex gamma^wv ({vrg_r.channel.value}) done.")

    gchi_r_aux_sum = gchi_r_aux.sum_over_all_vn(config.sys.beta)
    del gchi_r_aux

    gchi_r_aux_sum = create_generalized_chi_with_shell_correction(gchi_r_aux_sum, gchi0, u_loc)
    logger.log_info(f"Updated local susceptibility chi^w ({gchi_r_aux_sum.channel.value}) with asymptotic correction.")

    return gamma_r, gchi_r_aux_sum, vrg_r, f_r


def get_loc_self_energy_vrg(
    vrg_dens: LocalFourPoint,
    vrg_magn: LocalFourPoint,
    gchi_dens_sum: LocalFourPoint,
    gchi_magn_sum: LocalFourPoint,
    g_loc: GreensFunction,
    u_loc: LocalInteraction,
) -> SelfEnergy:
    """
    Performs the local self-energy calculation using the Schwinger-Dyson equation,
    see Paul Worm's thesis, Eq. (3.70) and Anna Galler's Thesis, P. 76 ff.
    """
    # 1=i, 2=j, 3=k, 4=l, 7=o, 8=p
    g_wv = MFHelper.wn_slices_gen(g_loc.mat, config.box.niv_core, config.box.niw_core)
    inner = vrg_dens - vrg_dens @ u_loc.as_channel(SpinChannel.DENS) @ gchi_dens_sum
    inner -= vrg_magn - vrg_magn @ u_loc.as_channel(SpinChannel.MAGN) @ gchi_magn_sum
    inner = 0.5 * inner.to_full_niw_range()
    sigma_sum = -1.0 / config.sys.beta * u_loc.times("kjop,ilpowv,lkwv->ijv", inner, g_wv)
    hartree_fock = u_loc.as_channel(SpinChannel.DENS).times("abcd,dc->ab", config.sys.occ)[..., None]

    return SelfEnergy((hartree_fock + sigma_sum)[None, None, None, ...])


def perform_local_schwinger_dyson(
    g_loc: GreensFunction, g2_dens: LocalFourPoint, g2_magn: LocalFourPoint, u_loc: LocalInteraction
):
    """
    Performs the local Schwinger-Dyson equation calculation for the local (DMFT) self-energy for sanity checks.
    Includes the calculation of the three-leg and full vertices, (auxiliary/bare/physical) susceptibilities
    and the irreducible vertices. Employs explicit asymptotics as proposed by
    Motoharu Kitatani et al. 2022 J. Phys. Mater. 5 034005; DOI 10.1088/2515-7639/ac7e6d.
    """
    gchi0 = create_generalized_chi0(g_loc)
    gchi0_inv_core = gchi0.cut_niv(config.box.niv_core).invert()

    gamma_dens, gchi_dens_sum, vrg_dens, f_dens = create_vertex_functions(g2_dens, gchi0, gchi0_inv_core, g_loc, u_loc)
    gamma_magn, gchi_magn_sum, vrg_magn, f_magn = create_vertex_functions(g2_magn, gchi0, gchi0_inv_core, g_loc, u_loc)

    sigma = get_loc_self_energy_vrg(vrg_dens, vrg_magn, gchi_dens_sum, gchi_magn_sum, g_loc, u_loc)
    config.logger.log_info("Self-energy Sigma^v done.")

    # This is saved since it is needed for the double-counting correction in the non-local routine
    (f_dens + 3 * f_magn).save(name="f_1dens_3magn", output_dir=config.output.output_path)

    return gamma_dens, gamma_magn, gchi_dens_sum, gchi_magn_sum, vrg_dens, vrg_magn, f_dens, f_magn, sigma


# ----------------------------------------------- AbinitioDGA algorithms -----------------------------------------------


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


def perform_local_schwinger_dyson_abinitio_dga(
    g_loc: GreensFunction,
    g2_dens: LocalFourPoint,
    g2_magn: LocalFourPoint,
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

    gchi0_loc_full = create_generalized_chi0(g_loc)
    logger.log_info("Local bare susceptibility chi_0^wv done.")
    gchi0_core = gchi0_loc_full.cut_niv(config.box.niv_core)

    # 1 + chi0 * F_r = gchi_r * (chi0)^(-1) = 1 + gamma_r or
    # F_r = -beta^2 * [chi0^(-1) - chi0^(-1) chi_r chi0^(-1)]
    # gamma_r is NOT the irreducible vertex in channel r but rather the three-point vertex from AbinitioDGA
    gchi0_inv_core = gchi0_core.invert()
    f_dens_loc = -config.sys.beta**2 * (gchi0_inv_core - gchi0_inv_core @ gchi_dens_loc @ gchi0_inv_core)
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
