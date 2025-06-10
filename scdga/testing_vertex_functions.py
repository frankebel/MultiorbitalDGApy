import os

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    oneband_folder = f"/home/julpe/Documents/DATA/Singleorb-DATA/N085_B12_5_low_iter/LDGA_Nk256_Nq256_wc60_vc40_vs20/"

    twoband_folder = (
        f"/home/julpe/Documents/DATA/Multiorb-DATA/oneband_as_twoband_diagonal/LDGA_Nk256_Nq256_wc60_vc40_vs20"
    )

    g2_dens_1 = np.load(os.path.join(oneband_folder, "g2_dens_loc.npy"))[0, 0, 0, 0]
    g2_dens_2 = np.load(os.path.join(twoband_folder, "g2_dens_loc.npy"))[0, 0, 0, 0]

    g2_magn_1 = np.load(os.path.join(oneband_folder, "g2_magn_loc.npy"))[0, 0, 0, 0]
    g2_magn_2 = np.load(os.path.join(twoband_folder, "g2_magn_loc.npy"))[0, 0, 0, 0]

    gchi_dens_1 = np.load(os.path.join(oneband_folder, "gchi_dens_loc.npy"))[0, 0, 0, 0]
    gchi_dens_2 = np.load(os.path.join(twoband_folder, "gchi_dens_loc.npy"))[0, 0, 0, 0]

    gchi_magn_1 = np.load(os.path.join(oneband_folder, "gchi_magn_loc.npy"))[0, 0, 0, 0]
    gchi_magn_2 = np.load(os.path.join(twoband_folder, "gchi_magn_loc.npy"))[0, 0, 0, 0]

    chi_dens_1 = np.load(os.path.join(oneband_folder, "chi_dens_loc.npy"))[0, 0, 0, 0]
    chi_dens_2 = np.load(os.path.join(twoband_folder, "chi_dens_loc.npy"))[0, 0, 0, 0]

    chi_magn_1 = np.load(os.path.join(oneband_folder, "chi_magn_loc.npy"))[0, 0, 0, 0]
    chi_magn_2 = np.load(os.path.join(twoband_folder, "chi_magn_loc.npy"))[0, 0, 0, 0]

    f_dens_1 = np.load(os.path.join(oneband_folder, "f_dens_loc.npy"))[0, 0, 0, 0]
    f_dens_2 = np.load(os.path.join(twoband_folder, "f_dens_loc.npy"))[0, 0, 0, 0]

    f_magn_1 = np.load(os.path.join(oneband_folder, "f_magn_loc.npy"))[0, 0, 0, 0]
    f_magn_2 = np.load(os.path.join(twoband_folder, "f_magn_loc.npy"))[0, 0, 0, 0]

    vrg_dens_1 = np.load(os.path.join(oneband_folder, "vrg_dens_loc.npy"))[0, 0, 0, 0]
    vrg_dens_2 = np.load(os.path.join(twoband_folder, "vrg_dens_loc.npy"))[0, 0, 0, 0]

    vrg_magn_1 = np.load(os.path.join(oneband_folder, "vrg_magn_loc.npy"))[0, 0, 0, 0]
    vrg_magn_2 = np.load(os.path.join(twoband_folder, "vrg_magn_loc.npy"))[0, 0, 0, 0]

    chi_phys_q_dens_1 = np.load(os.path.join(oneband_folder, "chi_phys_q_dens.npy"))[:, 0, 0, 0, 0, ...]
    chi_phys_q_dens_2 = np.load(os.path.join(twoband_folder, "chi_phys_q_dens.npy"))[:, 0, 0, 0, 0, ...]

    chi_phys_q_magn_1 = np.load(os.path.join(oneband_folder, "chi_phys_q_magn.npy"))[:, 0, 0, 0, 0, ...]
    chi_phys_q_magn_2 = np.load(os.path.join(twoband_folder, "chi_phys_q_magn.npy"))[:, 0, 0, 0, 0, ...]

    gamma_dens_1 = np.load(os.path.join(oneband_folder, "gamma_dens_loc.npy"))[0, 0, 0, 0]
    gamma_dens_2 = np.load(os.path.join(twoband_folder, "gamma_dens_loc.npy"))[0, 0, 0, 0]

    gamma_magn_1 = np.load(os.path.join(oneband_folder, "gamma_magn_loc.npy"))[0, 0, 0, 0]
    gamma_magn_2 = np.load(os.path.join(twoband_folder, "gamma_magn_loc.npy"))[0, 0, 0, 0]

    siw_dga_local_1 = np.load(os.path.join(oneband_folder, "siw_dga_local.npy"))[0, 0, 0, 0, 0]
    siw_dga_local_2 = np.load(os.path.join(twoband_folder, "siw_dga_local.npy"))[0, 0, 0, 0, 0]

    f_irrq_dens_pp_1 = np.load(os.path.join(oneband_folder, "f_irrq_dens_pp.npy"))[:, 0, 0, 0, 0, ...]
    f_irrq_dens_pp_2 = np.load(os.path.join(twoband_folder, "f_irrq_dens_pp.npy"))[:, 0, 0, 0, 0, ...]

    f_irrq_magn_pp_1 = np.load(os.path.join(oneband_folder, "f_irrq_magn_pp.npy"))[:, 0, 0, 0, 0, ...]
    f_irrq_magn_pp_2 = np.load(os.path.join(twoband_folder, "f_irrq_magn_pp.npy"))[:, 0, 0, 0, 0, ...]

    gamma_irrq_sing_pp_1 = np.load(os.path.join(oneband_folder + "/Eliashberg", "gamma_irrq_sing_pp.npy"))[
        :, 0, 0, 0, 0, ...
    ]
    gamma_irrq_sing_pp_2 = np.load(os.path.join(twoband_folder + "/Eliashberg", "gamma_irrq_sing_pp.npy"))[
        :, 0, 0, 0, 0, ...
    ]

    gamma_irrq_trip_pp_1 = np.load(os.path.join(oneband_folder + "/Eliashberg", "gamma_irrq_trip_pp.npy"))[
        :, 0, 0, 0, 0, ...
    ]
    gamma_irrq_trip_pp_2 = np.load(os.path.join(twoband_folder + "/Eliashberg", "gamma_irrq_trip_pp.npy"))[
        :, 0, 0, 0, 0, ...
    ]

    siw_dmft_1 = np.load(os.path.join(oneband_folder, "sigma_dmft.npy"))[0, 0, 0, 0, 0]
    siw_dmft_2 = np.load(os.path.join(twoband_folder, "sigma_dmft.npy"))[0, 0, 0, 0, 0]
    niv_dmft = siw_dmft_1.shape[-1] // 2
    siw_dmft_1 = siw_dmft_1[..., niv_dmft - 40 : niv_dmft + 40]
    siw_dmft_2 = siw_dmft_2[..., niv_dmft - 40 : niv_dmft + 40]

    delta_sigma_1 = siw_dmft_1 - siw_dga_local_1
    delta_sigma_2 = siw_dmft_2 - siw_dga_local_2

    giwk_dga_1 = np.load(os.path.join(oneband_folder, "giwk_dga.npy"))[0, 0, 0, 0, 0]
    giwk_dga_2 = np.load(os.path.join(twoband_folder, "giwk_dga.npy"))[0, 0, 0, 0, 0]

    siwk_dga_1 = np.load(os.path.join(oneband_folder, "sigma_dga.npy"))[0, 0, 0, 0, 0, niv_dmft - 40 : niv_dmft + 40]
    siwk_dga_2 = np.load(os.path.join(twoband_folder, "sigma_dga.npy"))[0, 0, 0, 0, 0, niv_dmft - 40 : niv_dmft + 40]

    plt.figure()
    # plt.plot(giwk_dga_1.real - giwk_dga_2.real, label="real, giwk")
    # plt.plot(giwk_dga_1.imag - giwk_dga_2.imag, label="imag, giwk")
    plt.plot(siwk_dga_1.real - siwk_dga_2.real, label="real, siwk")
    plt.plot(siwk_dga_1.imag - siwk_dga_2.imag, label="imag, siwk")
    plt.legend()
    plt.show()
