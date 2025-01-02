import logging
import os

import config
import w2dyn_aux
from local_four_point import LocalFourPoint
from local_greens_function import LocalGreensFunction
from local_self_energy import LocalSelfEnergy
from n_point_base import *


def load_from_w2dyn_file_and_update_config() -> (LocalGreensFunction, LocalSelfEnergy, LocalFourPoint, LocalFourPoint):
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
        config.sys.n = np.sum(np.diag(file.get_occ()[0, :, 0, :]))  # band spin band spin

    giw_spin_mean = np.mean(file.get_giw(), axis=1)
    giw = LocalGreensFunction(np.einsum("i...,ij->ij...", giw_spin_mean, np.eye(config.sys.n_bands)))
    siw_spin_mean = np.mean(file.get_siw(), axis=1)
    siw = LocalSelfEnergy(np.einsum("i...,ij->ij...", siw_spin_mean, np.eye(config.sys.n_bands)))

    file.close()

    file = w2dyn_aux.W2dynG4iwFile(fname=str(os.path.join(config.dmft.input_path, config.dmft.fname_2p)))
    g2_dens = LocalFourPoint(
        file.read_g2_full_multiband(config.sys.n_bands, channel=Channel.DENS), channel=Channel.DENS
    )
    g2_magn = LocalFourPoint(
        file.read_g2_full_multiband(config.sys.n_bands, channel=Channel.MAGN), channel=Channel.MAGN
    )
    file.close()

    return giw, siw, g2_dens, g2_magn


def update_frequency_boxes(niv: int, niw: int) -> None:
    logger = logging.getLogger()
    if config.box.niv == -1:
        config.box.niv = niv
        logger.info(f"Number of fermionic Matsubara frequency is set to '-1'. Using niv = {niv}.")
    elif config.box.niv > niv:
        config.box.niv = niv
        logger.info(
            f"Number of fermionic Matsubara frequencies cannot exceed available "
            f"frequencies in the DMFT four-point object. Using niv = {niv}."
        )

    if config.box.niw == -1:
        config.box.niw = niw
        logger.info(f"Number of bosonic Matsubara frequency is set to '-1'. Using niw = {niw}.")
    elif config.box.niw > niw:
        config.box.niw = niw
        logger.info(
            f"Number of bosonic Matsubara frequencies cannot exceed available "
            f"frequencies in the DMFT four-point object. Using niw = {niw}."
        )


def update_g2_from_dmft(g2_dens: LocalFourPoint, g2_magn: LocalFourPoint) -> (LocalFourPoint, LocalFourPoint):
    g2_dens = g2_dens.cut_niw_and_niv(config.box.niw, config.box.niv)
    g2_magn = g2_magn.cut_niw_and_niv(config.box.niw, config.box.niv)
    if config.dmft.do_sym_v_vp:
        g2_dens = g2_dens.symmetrize_v_vp()
        g2_magn = g2_magn.symmetrize_v_vp()
    return g2_dens, g2_magn
