import os

import matplotlib.pyplot as plt
import numpy as np


def show_diff(oneband, twoband, omega=0, title=""):
    oneband = oneband[omega]
    twoband = twoband[omega]

    vmax_real = max(np.max(oneband.real), np.max(twoband.real))
    vmin_real = min(np.min(oneband.real), np.min(twoband.real))

    vmax_imag = max(np.max(oneband.imag), np.max(twoband.imag))
    vmin_imag = min(np.min(oneband.imag), np.min(twoband.imag))

    fig, axes = plt.subplots(2, 3, figsize=(10, 10))  # 2x3 grid

    im1 = axes[0, 0].matshow(oneband.real, cmap="RdBu", vmin=vmin_real, vmax=vmax_real)
    axes[0, 0].set_title(f"{title}, Real, oneband")

    im2 = axes[1, 0].matshow(oneband.imag, cmap="RdBu", vmin=vmin_imag, vmax=vmax_imag)
    axes[1, 0].set_title(f"{title}, Imag, oneband")

    im3 = axes[0, 1].matshow(twoband.real, cmap="RdBu", vmin=vmin_real, vmax=vmax_real)
    axes[0, 1].set_title(f"{title}, Real, twoband")

    im4 = axes[1, 1].matshow(twoband.imag, cmap="RdBu", vmin=vmin_imag, vmax=vmax_imag)
    axes[1, 1].set_title(f"{title}, Imag, twoband")

    im5 = axes[0, 2].matshow(np.abs(oneband.real - twoband.real), cmap="RdBu")
    axes[0, 2].set_title("Real, Difference")
    im6 = axes[1, 2].matshow(np.abs(oneband.imag - twoband.imag), cmap="RdBu")
    axes[1, 2].set_title("Imag, Difference")

    fig.colorbar(im1, ax=axes[0, 0], aspect=15, fraction=0.08, location="right", pad=0.05)
    fig.colorbar(im2, ax=axes[1, 0], aspect=15, fraction=0.08, location="right", pad=0.05)
    fig.colorbar(im3, ax=axes[0, 1], aspect=15, fraction=0.08, location="right", pad=0.05)
    fig.colorbar(im4, ax=axes[1, 1], aspect=15, fraction=0.08, location="right", pad=0.05)
    fig.colorbar(im5, ax=axes[0, 2], aspect=15, fraction=0.08, location="right", pad=0.05)
    fig.colorbar(im6, ax=axes[1, 2], aspect=15, fraction=0.08, location="right", pad=0.05)

    # Adjust layout
    fig.suptitle(title)
    fig.tight_layout()
    fig.legend()
    plt.grid()
    fig.show()


if __name__ == "__main__":
    oneband_folder = f"/home/julpe/Documents/DATA/Singleorb-DATA/N085/LDGA_Nk256_Nq256_wc60_vc40_vs0/"
    twoband_folder = f"/home/julpe/Documents/DATA/Multiorb-DATA/oneband_as_twoband_diagonal_higher_stat_for_vertex_2/LDGA_Nk256_Nq256_wc60_vc40_vs0/"

    # twoband_folder = (
    #    f"/home/julpe/Documents/DATA/Multiorb-DATA/oneband_as_twoband_diagonal/LDGA_Nk256_Nq256_wc60_vc40_vs0/"
    # )

    g2_dens_1 = np.load(os.path.join(oneband_folder, "g2_dens_loc.npy"))[0, 0, 0, 0]
    g2_dens_2 = np.load(os.path.join(twoband_folder, "g2_dens_loc.npy"))[0, 0, 0, 0]

    g2_magn_1 = np.load(os.path.join(oneband_folder, "g2_magn_loc.npy"))[0, 0, 0, 0]
    g2_magn_2 = np.load(os.path.join(twoband_folder, "g2_magn_loc.npy"))[0, 0, 0, 0]

    show_diff(g2_dens_1, g2_dens_2, 0, "G2_dens")
    show_diff(g2_magn_1, g2_magn_2, 0, "G2_magn")

    gchi_dens_1 = np.load(os.path.join(oneband_folder, "gchi_dens_loc.npy"))[0, 0, 0, 0]
    gchi_dens_2 = np.load(os.path.join(twoband_folder, "gchi_dens_loc.npy"))[0, 0, 0, 0]

    gchi_magn_1 = np.load(os.path.join(oneband_folder, "gchi_magn_loc.npy"))[0, 0, 0, 0]
    gchi_magn_2 = np.load(os.path.join(twoband_folder, "gchi_magn_loc.npy"))[0, 0, 0, 0]

    show_diff(gchi_dens_1, gchi_dens_2, 0, "Gchi_dens")
    show_diff(gchi_magn_1, gchi_magn_2, 0, "Gchi_magn")

    chi_dens_1 = np.load(os.path.join(oneband_folder, "chi_dens_loc.npy"))[0, 0, 0, 0]
    chi_dens_2 = np.load(os.path.join(twoband_folder, "chi_dens_loc.npy"))[0, 0, 0, 0]

    chi_magn_1 = np.load(os.path.join(oneband_folder, "chi_magn_loc.npy"))[0, 0, 0, 0]
    chi_magn_2 = np.load(os.path.join(twoband_folder, "chi_magn_loc.npy"))[0, 0, 0, 0]

    f_dens_1 = np.load(os.path.join(oneband_folder, "f_dens_loc.npy"))[0, 0, 0, 0]
    f_dens_2 = np.load(os.path.join(twoband_folder, "f_dens_loc.npy"))[0, 0, 0, 0]

    f_magn_1 = np.load(os.path.join(oneband_folder, "f_magn_loc.npy"))[0, 0, 0, 0]
    f_magn_2 = np.load(os.path.join(twoband_folder, "f_magn_loc.npy"))[0, 0, 0, 0]

    show_diff(f_dens_1, f_dens_2, 0, "F_dens")
    show_diff(f_magn_1, f_magn_2, 0, "F_magn")

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

    show_diff(gamma_dens_1, gamma_dens_2, 0, "Gamma_dens")
    show_diff(gamma_magn_1, gamma_magn_2, 0, "Gamma_magn")

    gchi_dens_inverted_1 = np.load(os.path.join(oneband_folder, "gchi_dens_inverted.npy"))[0, 0, 0, 0]
    gchi_dens_inverted_2 = np.load(os.path.join(twoband_folder, "gchi_dens_inverted.npy"))[0, 0, 0, 0]

    gchi_magn_inverted_1 = np.load(os.path.join(oneband_folder, "gchi_magn_inverted.npy"))[0, 0, 0, 0]
    gchi_magn_inverted_2 = np.load(os.path.join(twoband_folder, "gchi_magn_inverted.npy"))[0, 0, 0, 0]

    show_diff(gchi_dens_inverted_1, gchi_dens_inverted_2, 0, "Gchi_dens Inverted")
    show_diff(gchi_magn_inverted_1, gchi_magn_inverted_2, 0, "Gchi_magn Inverted")

    chi_tilde_core_inv_1 = np.load(os.path.join(oneband_folder, "chi_tilde_core_inv.npy"))[0, 0, 0, 0]
    chi_tilde_core_inv_2 = np.load(os.path.join(twoband_folder, "chi_tilde_core_inv.npy"))[0, 0, 0, 0]

    show_diff(
        chi_tilde_core_inv_1,
        chi_tilde_core_inv_2,
        0,
        "Chi Tilde Core Inverted",
    )

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

    exit()

    plt.figure()
    # plt.plot(giwk_dga_1.real - giwk_dga_2.real, label="real, giwk")
    # plt.plot(giwk_dga_1.imag - giwk_dga_2.imag, label="imag, giwk")
    plt.plot(siwk_dga_1.real - siwk_dga_2.real, label="real, siwk")
    plt.plot(siwk_dga_1.imag - siwk_dga_2.imag, label="imag, siwk")
    plt.legend()
    plt.show()
