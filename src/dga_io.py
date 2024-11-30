import logging
import os

import numpy as np

import config
import w2dyn_aux
from local_four_point import LocalFourPoint
from local_two_point import LocalTwoPoint
from local_self_energy import LocalSelfEnergy
from local_greens_function import LocalGreensFunction


def load_from_w2dyn_file():
    dmft_input = {}

    file = w2dyn_aux.W2dynFile(fname=os.path.join(config.input_path, config.dmft_1p_filename))

    # TODO: EXTEND TO MULTIORBITAL DATA ONCE I HAVE THE INPUT FILES

    dmft_input["n"] = file.get_totdens()
    if dmft_input["n"] == 0:
        dmft_input["n"] = np.sum(np.diag(file.get_occupation()[0, :, 0, :]))  # band spin band spin
    dmft_input["beta"] = file.get_beta()
    dmft_input["u"] = file.get_udd()
    dmft_input["mu"] = file.get_mu()

    config.beta = dmft_input["beta"]
    config.n = dmft_input["n"]
    config.mu = dmft_input["mu"]

    dmft_input["giw"] = LocalGreensFunction.create_from_dmft(np.mean(file.get_giw(), axis=1))  # band spin niv
    dmft_input["siw"] = LocalSelfEnergy.create_from_dmft(np.mean(file.get_siw(), axis=1))

    file.close()

    if config.dmft_2p_filename is not None:
        file = w2dyn_aux.W2dynG4iwFile(fname=os.path.join(config.input_path, config.dmft_2p_filename))
        dmft_input["g4iw_dens"] = LocalFourPoint(file.read_g2_full(channel="dens"), channel="dens")
        dmft_input["g4iw_magn"] = LocalFourPoint(file.read_g2_full(channel="magn"), channel="magn")

        if len(dmft_input["g4iw_dens"].current_shape) == 3:
            dmft_input["g4iw_dens"].mat = dmft_input["g4iw_dens"].mat.reshape(
                (1,) * 4 + dmft_input["g4iw_dens"].mat.shape
            )
            dmft_input["g4iw_magn"].mat = dmft_input["g4iw_magn"].mat.reshape(
                (1,) * 4 + dmft_input["g4iw_magn"].mat.shape
            )

        file.close()

    return dmft_input


def update_frequency_boxes(niv: int, niw: int):
    logger = logging.getLogger()
    if config.niv == -1:
        config.niv = niv
        logger.info(f"Number of fermionic Matsubara frequency is set to '-1'. Taking niv = {niv}.")
    elif config.niv > niv:
        config.niv = niv
        logger.info(
            "Number of fermionic Matsubara frequencies cannot exceed available frequencies in the DMFT four-point object."
        )

    if config.niw == -1:
        config.niw = niw
        logger.info(f"Number of bosonic Matsubara frequency is set to '-1'. Taking niw = {niw}.")
    elif config.niw > niw:
        config.niw = niw
        logger.info(
            "Number of bosonic Matsubara frequencies cannot exceed available frequencies in the DMFT four-point object."
        )
