import os

import numpy as np

from scdga.four_point import FourPoint
from scdga.n_point_base import SpinChannel

if __name__ == "__main__":
    my_folder = "/home/julpe/Documents/DATA/Singleorb-DATA/N085/LDGA_Nk64_Nq64_wc60_vc40_vs0"
    pauls_folder = "/home/julpe/Documents/DATA/Singleorb-DATA/N085/LDGA_none_Nk64_Nq64_wc60_vc40_vs0"

    my_folder = os.path.join(my_folder, "Eliashberg")

    my_full_vertex_dens = (
        FourPoint.load(f"{my_folder}/f_dens_irrq.npy", SpinChannel.DENS).to_full_niw_range().mat[:, 0, 0, 0, 0, ...]
    )
    my_full_vertex_magn = (
        FourPoint.load(f"{my_folder}/f_magn_irrq.npy", SpinChannel.MAGN).to_full_niw_range().mat[:, 0, 0, 0, 0, ...]
    )
    # my_full_vertex_dens_1 = (
    # FourPoint.load(f"{my_folder}/f_1_q_dens.npy", SpinChannel.DENS).to_full_niw_range().mat[:, 0, 0, 0, 0, ...]
    # )
    # my_full_vertex_magn_1 = (
    #    FourPoint.load(f"{my_folder}/f_1_q_magn.npy", SpinChannel.MAGN).to_full_niw_range().mat[:, 0, 0, 0, 0, ...]
    # )
    # my_full_vertex_dens_2 = (
    #   FourPoint.load(f"{my_folder}/f_2_q_dens.npy", SpinChannel.DENS).to_full_niw_range().mat[:, 0, 0, 0, 0, ...]
    # )
    # my_full_vertex_magn_2 = (
    #   FourPoint.load(f"{my_folder}/f_2_q_magn.npy", SpinChannel.MAGN).to_full_niw_range().mat[:, 0, 0, 0, 0, ...]
    # )

    paul_full_vertex_dens = np.load(f"{pauls_folder}/F_dens.npy")
    paul_full_vertex_magn = np.load(f"{pauls_folder}/F_magn.npy")
    # paul_full_vertex_dens_1 = np.load(f"{pauls_folder}/F_dens_1.npy")
    # paul_full_vertex_magn_1 = np.load(f"{pauls_folder}/F_magn_1.npy")
    # paul_full_vertex_dens_2 = np.load(f"{pauls_folder}/F_dens_2.npy")
    # paul_full_vertex_magn_2 = np.load(f"{pauls_folder}/F_magn_2.npy")

    abs_diff_dens = np.max(np.abs(my_full_vertex_dens - paul_full_vertex_dens))
    abs_diff_magn = np.max(np.abs(my_full_vertex_magn - paul_full_vertex_magn))
    # abs_diff_dens_1 = np.max(np.abs(my_full_vertex_dens_1 - paul_full_vertex_dens_1))
    # abs_diff_magn_1 = np.max(np.abs(my_full_vertex_magn_1 - paul_full_vertex_magn_1))
    # abs_diff_dens_2 = np.max(np.abs(my_full_vertex_dens_2 - paul_full_vertex_dens_2))
    # abs_diff_magn_2 = np.max(np.abs(my_full_vertex_magn_2 - paul_full_vertex_magn_2))

    print("Hellow")
