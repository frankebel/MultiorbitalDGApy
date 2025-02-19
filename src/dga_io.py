import os

import brillouin_zone as bz
import config
import w2dyn_aux
from hamiltonian import Hamiltonian
from local_four_point import LocalFourPoint
from greens_function import GreensFunction
from self_energy import SelfEnergy
from n_point_base import *


def _uniquify_path(path: str = None):
    """
    path: path to be checked for uniqueness
    return: updated unique path
    """
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "_" + str(counter) + extension
        counter += 1

    return path


def load_from_w2dyn_file_and_update_config() -> tuple[GreensFunction, SelfEnergy, LocalFourPoint, LocalFourPoint]:
    file = w2dyn_aux.W2dynFile(fname=str(os.path.join(config.dmft.input_path, config.dmft.fname_1p)))

    config.sys.beta = file.get_beta()

    config.lattice.interaction.udd = file.get_udd()
    config.lattice.interaction.udp = file.get_udp()
    config.lattice.interaction.upp = file.get_upp()
    config.lattice.interaction.uppod = file.get_uppod()
    config.lattice.interaction.jdd = file.get_jdd()
    config.lattice.interaction.jdp = file.get_jdp()
    config.lattice.interaction.jpp = file.get_jpp()
    config.lattice.interaction.jppod = file.get_jppod()
    config.lattice.interaction.vdd = file.get_vdd()
    config.lattice.interaction.vpp = file.get_vpp()

    config.sys.mu = file.get_mu()

    config.sys.n_bands = file.get_nd() + file.get_np()
    config.sys.n = file.get_totdens()

    if config.sys.n == 0:
        config.sys.n = sum(
            np.sum(np.diag(file.get_occ()[i, :, i, :])) for i in range(config.sys.n_bands)
        )  # band spin band spin

    giw_spin_mean = np.mean(file.get_giw(), axis=1)
    giw = GreensFunction(np.einsum("i...,ij->ij...", giw_spin_mean, np.eye(config.sys.n_bands)))
    siw_spin_mean = np.mean(file.get_siw(), axis=1)
    siw = SelfEnergy(np.einsum("i...,ij->ij...", siw_spin_mean, np.eye(config.sys.n_bands)))

    file.close()

    file = w2dyn_aux.W2dynG4iwFile(fname=str(os.path.join(config.dmft.input_path, config.dmft.fname_2p)))
    g2_dens = LocalFourPoint(
        file.read_g2_full_multiband(config.sys.n_bands, channel=Channel.DENS), channel=Channel.DENS
    )
    g2_magn = LocalFourPoint(
        file.read_g2_full_multiband(config.sys.n_bands, channel=Channel.MAGN), channel=Channel.MAGN
    )
    file.close()

    config.lattice.hamiltonian = set_hamiltonian(
        config.lattice.type, config.lattice.er_input, config.lattice.interaction_type, config.lattice.interaction_input
    )

    _update_frequency_boxes(g2_dens.niw, g2_dens.niv)

    output_format = "LDGA_Nk{}_Nq{}_wc{}_vc{}_vs{}".format(
        config.lattice.k_grid.nk_tot,
        config.lattice.q_grid.nk_tot,
        config.box.niw,
        config.box.niv,
        config.box.niv_asympt,
    )
    config.output.output_path = _uniquify_path(os.path.join(config.output.output_path, output_format))

    if not os.path.exists(config.output.output_path):
        os.makedirs(config.output.output_path)

    g2_dens, g2_magn = _update_g2_from_dmft(g2_dens, g2_magn)

    return giw, siw, g2_dens, g2_magn


def _update_frequency_boxes(niw: int, niv: int) -> None:
    logger = config.logger
    if config.box.niv == -1:
        config.box.niv = niv
        logger.log_info(f"Number of fermionic Matsubara frequency is set to '-1'. Using niv = {niv}.")
    elif config.box.niv > niv:
        config.box.niv = niv
        logger.log_info(
            f"Number of fermionic Matsubara frequencies cannot exceed available "
            f"frequencies in the DMFT four-point object. Using niv = {niv}."
        )

    if config.box.niw == -1:
        config.box.niw = niw
        logger.log_info(f"Number of bosonic Matsubara frequency is set to '-1'. Using niw = {niw}.")
    elif config.box.niw > niw:
        config.box.niw = niw
        logger.log_info(
            f"Number of bosonic Matsubara frequencies cannot exceed available "
            f"frequencies in the DMFT four-point object. Using niw = {niw}."
        )

    config.box.niv_full = config.box.niv + config.box.niv_asympt


def _update_g2_from_dmft(g2_dens: LocalFourPoint, g2_magn: LocalFourPoint) -> (LocalFourPoint, LocalFourPoint):
    g2_dens = g2_dens.cut_niw_and_niv(config.box.niw, config.box.niv)
    g2_magn = g2_magn.cut_niw_and_niv(config.box.niw, config.box.niv)
    if config.dmft.do_sym_v_vp:
        config.logger.log_info("Symmetrizing G2_dens and G2_magn with respect to v and v'.")
        g2_dens = g2_dens.symmetrize_v_vp()
        g2_magn = g2_magn.symmetrize_v_vp()
    return g2_dens, g2_magn


def set_hamiltonian(er_type: str, er_input: str | list, int_type: str, int_input: str | list) -> Hamiltonian:
    """
    Sets the Hamiltonian based on the input from the config file. \n
    The kinetic part can be set in two ways: \n
    1. By providing the single-band hopping parameters t, tp, tpp. \n
    2. By providing the path + filename to the wannier_hr / wannier_hk file. \n
    The interaction can be set in three ways: \n
    1. By retrieving the data from the DMFT files. \n
    2. By providing the Kanamori interaction parameters [n_bands, U, J, (V)]. \n
    3. By providing the full path + filename to the U-matrix file. \n
    """
    ham = Hamiltonian()
    if er_type == "t_tp_tpp":
        if not isinstance(er_input, list):
            raise ValueError("Invalid input for t, tp, tpp.")
        ham = ham.kinetic_one_band_2d_t_tp_tpp(*er_input)
    elif er_type == "from_wannier90":
        if not isinstance(er_input, str):
            raise ValueError("Invalid input for wannier_hr.dat.")
        ham = ham.read_er_w2k(er_input)
    elif er_type == "from_wannierHK":
        # ATTENTION: currently this is only implemented for 2D square systems
        if not isinstance(er_input, str):
            raise ValueError("Invalid input for wannier.hk.")
        ham, k_grid = ham.read_hk_w2k(er_input)
        if k_grid is not None:
            config.logger.log_info("Using q- and k-grid from wannier.hk file.")
            config.lattice.nk = ham.get_ek().shape[:3]
            config.lattice.nq = config.lattice.nk
            config.lattice.k_grid = bz.KGrid(config.lattice.nk, config.lattice.symmetries)
            config.lattice.q_grid = bz.KGrid(config.lattice.nq, config.lattice.symmetries)
    else:
        raise NotImplementedError(f"Hamiltonian type {er_type} not supported.")

    if int_type == "local_from_dmft" or int_type == "" or int_type is None:
        return ham.single_band_interaction(config.lattice.interaction.udd)
    elif int_type == "kanamori_from_dmft":
        return ham.kanamori_interaction(
            config.sys.n_bands,
            config.lattice.interaction.udd,
            config.lattice.interaction.jdd,
            config.lattice.interaction.vdd,
        )
    elif int_type == "kanamori":
        if not isinstance(int_input, list) or not 3 <= len(int_input) <= 4:
            raise ValueError("Invalid input for kanamori interaction.")
        return ham.kanamori_interaction(*int_input)
    elif int_type == "custom":
        if not isinstance(int_input, str):
            raise ValueError("Invalid input for umatrix file.")
        return ham.read_umatrix(int_input)
    else:
        raise NotImplementedError(f"Interaction type {int_type} not supported.")
