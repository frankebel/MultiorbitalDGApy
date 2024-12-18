import logging
import os

import numpy as np

import config
import w2dyn_aux
from i_have_channel import Channel
from local_four_point import LocalFourPoint
from local_greens_function import LocalGreensFunction
from local_self_energy import LocalSelfEnergy


def load_from_w2dyn_file_and_update_config() -> (LocalGreensFunction, LocalSelfEnergy, LocalFourPoint, LocalFourPoint):
    file = w2dyn_aux.W2dynFile(fname=os.path.join(config.input_path, config.dmft_1p_filename))
    # TODO: EXTEND TO MULTIORBITAL DATA ONCE I HAVE THE INPUT FILES

    config.n_dmft = file.get_totdens()
    if config.n_dmft == 0:
        config.n_dmft = np.sum(np.diag(file.get_occ()[0, :, 0, :]))  # band spin band spin
    config.beta = file.get_beta()

    config.interaction.udd = file.get_udd()
    config.interaction.udp = file.get_udp()
    config.interaction.upp = file.get_upp()
    config.interaction.uppod = file.get_uppod()
    config.interaction.jdd = file.get_jdd()
    config.interaction.jdp = file.get_jdp()
    config.interaction.jpp = file.get_jpp()
    config.interaction.jppod = file.get_jppod()
    config.interaction.vdd = file.get_vdd()
    config.interaction.vpp = file.get_vpp()

    config.n_bands = file.get_nd() + file.get_np()

    config.mu = file.get_mu()

    giw = LocalGreensFunction.from_dmft(np.mean(file.get_giw(), axis=1))
    siw = LocalSelfEnergy.from_dmft(np.mean(file.get_siw(), axis=1))
    file.close()

    file = w2dyn_aux.W2dynG4iwFile(fname=os.path.join(config.input_path, config.dmft_2p_filename))
    g2_dens = LocalFourPoint(file.read_g2_full(channel=Channel.DENS), channel=Channel.DENS)
    g2_magn = LocalFourPoint(file.read_g2_full(channel=Channel.MAGN), channel=Channel.MAGN)

    if len(g2_dens.current_shape) == 3:
        g2_dens.mat = g2_dens.mat.reshape((1,) * 4 + g2_dens.mat.shape)
    if len(g2_magn.current_shape) == 3:
        g2_magn.mat = g2_magn.mat.reshape((1,) * 4 + g2_magn.mat.shape)
    file.close()

    return giw, siw, g2_dens, g2_magn


def update_frequency_boxes(niv: int, niw: int) -> None:
    logger = logging.getLogger()
    if config.niv == -1:
        config.niv = niv
        logger.info(f"Number of fermionic Matsubara frequency is set to '-1'. Taking niv = {niv}.")
    elif config.niv > niv:
        config.niv = niv
        logger.info(
            f"Number of fermionic Matsubara frequencies cannot exceed available "
            f"frequencies in the DMFT four-point object. Taking niv = {niv}."
        )

    if config.niw == -1:
        config.niw = niw
        logger.info(f"Number of bosonic Matsubara frequency is set to '-1'. Taking niw = {niw}.")
    elif config.niw > niw:
        config.niw = niw
        logger.info(
            f"Number of bosonic Matsubara frequencies cannot exceed available "
            f"frequencies in the DMFT four-point object. Taking niw = {niw}."
        )


def update_g2_from_dmft(g2_dens: LocalFourPoint, g2_magn: LocalFourPoint) -> (LocalFourPoint, LocalFourPoint):
    g2_dens = g2_dens.cut_niw_and_niv(config.niw, config.niv)
    g2_magn = g2_magn.cut_niw_and_niv(config.niw, config.niv)
    if config.do_sym_v_vp:
        g2_dens = g2_dens.symmetrize_v_vp()
        g2_magn = g2_magn.symmetrize_v_vp()
    return g2_dens, g2_magn
