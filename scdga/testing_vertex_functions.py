import os

import matplotlib.pyplot as plt
import numpy as np

from scdga.hamiltonian import Hamiltonian


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

    fig.suptitle(title)
    fig.tight_layout()
    fig.legend()
    plt.grid()
    fig.show()


if __name__ == "__main__":
    oneband_folder = f"/home/julpe/Documents/DATA/Singleorb-DATA/N085/LDGA_Nk256_Nq256_wc60_vc40_vs0/"
    twoband_folder = f"/home/julpe/Documents/DATA/Multiorb-DATA/oneband_as_twoband_diagonal_higher_stat_for_vertex_2/LDGA_Nk256_Nq256_wc60_vc40_vs0/"

    oneband_folder_eliashberg = oneband_folder + "Eliashberg/"
    twoband_folder_eliashberg = twoband_folder + "Eliashberg/"

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

    vrg_dens_1 = np.load(os.path.join(oneband_folder, "vrg_dens_loc.npy"))[0, 0, 0, 0, 0]
    vrg_dens_2 = np.load(os.path.join(twoband_folder, "vrg_dens_loc.npy"))[0, 0, 0, 0, 0]

    vrg_magn_1 = np.load(os.path.join(oneband_folder, "vrg_magn_loc.npy"))[0, 0, 0, 0, 0]
    vrg_magn_2 = np.load(os.path.join(twoband_folder, "vrg_magn_loc.npy"))[0, 0, 0, 0, 0]

    plt.figure()
    plt.plot(vrg_dens_1 - vrg_dens_2, label="vrg_dens difference")
    plt.plot(vrg_magn_1 - vrg_magn_2, label="vrg_magn difference")
    plt.plot()
    plt.legend()
    plt.show()

    chi_phys_q_dens_1 = np.load(os.path.join(oneband_folder, "chi_phys_q_dens.npy"))[:, 0, 0, 0, 0, ...]
    chi_phys_q_dens_2 = np.load(os.path.join(twoband_folder, "chi_phys_q_dens.npy"))[:, 0, 0, 0, 0, ...]

    chi_phys_q_magn_1 = np.load(os.path.join(oneband_folder, "chi_phys_q_magn.npy"))[:, 0, 0, 0, 0, ...]
    chi_phys_q_magn_2 = np.load(os.path.join(twoband_folder, "chi_phys_q_magn.npy"))[:, 0, 0, 0, 0, ...]

    plt.figure()
    plt.plot(np.max(chi_phys_q_dens_1 - chi_phys_q_dens_2, axis=0), label="chi_phys_q_dens difference")
    plt.plot(np.max(chi_phys_q_magn_1 - chi_phys_q_magn_2, axis=0), label="chi_phys_q_magn difference")
    plt.plot()
    plt.legend()
    plt.show()

    gchi0_q_inv_1 = np.load(os.path.join(oneband_folder_eliashberg, "gchi0_q_inv_rank_0.npy"))[:, 0, 0, 0, 0, 0]
    gchi0_q_inv_2 = np.load(os.path.join(twoband_folder_eliashberg, "gchi0_q_inv_rank_0.npy"))[:, 0, 0, 0, 0, 0]

    plt.figure()
    plt.plot(np.sum(gchi0_q_inv_1 - gchi0_q_inv_2, axis=0), label="gchi0_inv difference")
    plt.plot()
    plt.legend()
    plt.show()

    gchi0_q_1 = np.load(os.path.join(oneband_folder, "gchi0_q_rank_0.npy"))[:, 0, 0, 0, 0, 0]
    gchi0_q_2 = np.load(os.path.join(twoband_folder, "gchi0_q_rank_0.npy"))[:, 0, 0, 0, 0, 0]

    plt.figure()
    plt.plot(np.sum(gchi0_q_1 - gchi0_q_2, axis=0), label="gchi0_q difference")
    plt.plot()
    plt.legend()
    plt.show()

    gchi0_1 = np.load(os.path.join(oneband_folder, "gchi0_loc.npy"))[0, 0, 0, 0, 0]
    gchi0_2 = np.load(os.path.join(twoband_folder, "gchi0_loc.npy"))[0, 0, 0, 0, 0]

    plt.figure()
    plt.plot(gchi0_1 - gchi0_2, label="gchi0 difference")
    plt.plot()
    plt.legend()
    plt.show()

    g_loc_1 = np.load(os.path.join(oneband_folder, "g_loc.npy"))[0, 0]
    g_loc_2 = np.load(os.path.join(twoband_folder, "g_loc.npy"))[0, 0]

    niv_dmft = g_loc_1.shape[-1] // 2
    g_loc_1 = g_loc_1[..., niv_dmft - 100 : niv_dmft + 100]
    g_loc_2 = g_loc_2[..., niv_dmft - 100 : niv_dmft + 100]

    g_dga_1 = np.load(os.path.join(oneband_folder, "g_dga.npy"))[..., 0, 0, :]
    g_dga_2 = np.load(os.path.join(twoband_folder, "g_dga.npy"))[..., 0, 0, :]

    plt.figure()
    plt.plot(g_loc_1 - g_loc_2, label="g_loc difference")
    plt.plot(np.max(g_dga_1 - g_dga_2, axis=(0, 1, 2)), label="g_dga difference")
    plt.plot()
    plt.legend()
    plt.show()

    gchi_aux_q_dens_rank_0_1 = np.load(os.path.join(oneband_folder_eliashberg, "gchi_aux_q_dens_rank_0.npy"))[
        :, 0, 0, 0, 0, ...
    ]
    gchi_aux_q_dens_rank_0_2 = np.load(os.path.join(twoband_folder_eliashberg, "gchi_aux_q_dens_rank_0.npy"))[
        :, 0, 0, 0, 0, ...
    ]

    gchi_aux_q_magn_rank_0_1 = np.load(os.path.join(oneband_folder_eliashberg, "gchi_aux_q_magn_rank_0.npy"))[
        :, 0, 0, 0, 0, ...
    ]
    gchi_aux_q_magn_rank_0_2 = np.load(os.path.join(twoband_folder_eliashberg, "gchi_aux_q_magn_rank_0.npy"))[
        :, 0, 0, 0, 0, ...
    ]

    show_diff(np.sum(gchi_aux_q_dens_rank_0_1, axis=0), np.sum(gchi_aux_q_dens_rank_0_2, axis=0), 0, "Gchi_aux_q_dens")
    show_diff(np.sum(gchi_aux_q_magn_rank_0_1, axis=0), np.sum(gchi_aux_q_magn_rank_0_2, axis=0), 0, "Gchi_aux_q_magn")

    gchi_aux_q_dens_sum_rank_0_1 = np.load(os.path.join(oneband_folder_eliashberg, "gchi_aux_q_dens_sum_rank_0.npy"))[
        :, 0, 0, 0, 0, ...
    ]
    gchi_aux_q_dens_sum_rank_0_2 = np.load(os.path.join(twoband_folder_eliashberg, "gchi_aux_q_dens_sum_rank_0.npy"))[
        :, 0, 0, 0, 0, ...
    ]

    gchi_aux_q_magn_sum_rank_0_1 = np.load(os.path.join(oneband_folder_eliashberg, "gchi_aux_q_magn_sum_rank_0.npy"))[
        :, 0, 0, 0, 0, ...
    ]
    gchi_aux_q_magn_sum_rank_0_2 = np.load(os.path.join(twoband_folder_eliashberg, "gchi_aux_q_magn_sum_rank_0.npy"))[
        :, 0, 0, 0, 0, ...
    ]

    plt.figure()
    plt.plot(
        np.sum(gchi_aux_q_dens_sum_rank_0_1 - gchi_aux_q_dens_sum_rank_0_2, axis=0),
        label="gchi_aux_q_dens_sum difference",
    )
    plt.plot(
        np.sum(gchi_aux_q_magn_sum_rank_0_1 - gchi_aux_q_magn_sum_rank_0_2, axis=0),
        label="gchi_aux_q_magn_sum difference",
    )
    plt.plot()
    plt.legend()
    plt.show()

    vrg_q_dens_rank_0_1 = np.load(os.path.join(oneband_folder_eliashberg, "vrg_q_dens_rank_0.npy"))[
        :, 0, 0, 0, 0, 0, ...
    ]
    vrg_q_dens_rank_0_2 = np.load(os.path.join(twoband_folder_eliashberg, "vrg_q_dens_rank_0.npy"))[
        :, 0, 0, 0, 0, 0, ...
    ]

    vrg_q_magn_rank_0_1 = np.load(os.path.join(oneband_folder_eliashberg, "vrg_q_magn_rank_0.npy"))[
        :, 0, 0, 0, 0, 0, ...
    ]
    vrg_q_magn_rank_0_2 = np.load(os.path.join(twoband_folder_eliashberg, "vrg_q_magn_rank_0.npy"))[
        :, 0, 0, 0, 0, 0, ...
    ]

    plt.figure()
    plt.plot(np.sum(vrg_q_dens_rank_0_1 - vrg_q_dens_rank_0_2, axis=0), label="vrg_q_dens difference")
    plt.plot(np.sum(vrg_q_magn_rank_0_1 - vrg_q_magn_rank_0_2, axis=0), label="vrg_q_magn difference")
    plt.plot()
    plt.legend()
    plt.show()

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

    giwk_dga_1_nonlocal = np.load(os.path.join(oneband_folder, "giwk_dga.npy"))[:, :, :, 0, 0]
    giwk_dga_2_nonlocal = np.load(os.path.join(twoband_folder, "giwk_dga.npy"))[:, :, :, 0, 0]

    plt.figure()
    plt.plot(np.max(giwk_dga_1_nonlocal.real - giwk_dga_2_nonlocal.real, axis=(0, 1, 2)), label="real, giwk, nonlocal")
    plt.plot(np.max(giwk_dga_1_nonlocal.imag - giwk_dga_2_nonlocal.imag, axis=(0, 1, 2)), label="imag, giwk, nonlocal")
    plt.legend()
    plt.show()

    g_dga_1_nonlocal = np.load(os.path.join(oneband_folder, "g_dga.npy"))[:, :, :, 0, 0]
    g_dga_2_nonlocal = np.load(os.path.join(twoband_folder, "g_dga.npy"))[:, :, :, 0, 0]

    plt.figure()
    plt.plot(np.max(g_dga_1_nonlocal.real - g_dga_2_nonlocal.real, axis=(0, 1, 2)), label="real, g dga, nonlocal")
    plt.plot(np.max(g_dga_1_nonlocal.imag - g_dga_2_nonlocal.imag, axis=(0, 1, 2)), label="imag, g dga, nonlocal")
    plt.legend()
    plt.show()

    giwk_dga_1 = np.load(os.path.join(oneband_folder, "giwk_dga.npy"))[0, 0, 0, 0, 0]
    giwk_dga_2 = np.load(os.path.join(twoband_folder, "giwk_dga.npy"))[0, 0, 0, 0, 0]

    siwk_dga_1 = np.load(os.path.join(oneband_folder, "sigma_dga.npy"))[0, 0, 0, 0, 0, niv_dmft - 40 : niv_dmft + 40]
    siwk_dga_2 = np.load(os.path.join(twoband_folder, "sigma_dga.npy"))[0, 0, 0, 0, 0, niv_dmft - 40 : niv_dmft + 40]

    plt.figure()
    plt.plot(giwk_dga_1.real - giwk_dga_2.real, label="real, giwk")
    plt.plot(giwk_dga_1.imag - giwk_dga_2.imag, label="imag, giwk")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(siwk_dga_1.real - siwk_dga_2.real, label="real, siwk")
    plt.plot(siwk_dga_1.imag - siwk_dga_2.imag, label="imag, siwk")
    plt.legend()
    plt.show()

    siwk_test_1 = 1.0 / g_loc_1 - 1.0 / giwk_dga_1
    siwk_test_2 = 1.0 / g_loc_2 - 1.0 / giwk_dga_2

    niv_test = siwk_test_1.shape[-1] // 2

    plt.figure()
    plt.plot(siwk_test_1.real - siwk_test_2.real, label="real, siwk")
    plt.plot(siwk_test_1.imag - siwk_test_2.imag, label="imag, siwk")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(siwk_dga_1.real - siwk_test_1[niv_test - 40 : niv_test + 40].real, label="real, difference, oneband")
    plt.plot(siwk_dga_1.imag - siwk_test_1[niv_test - 40 : niv_test + 40].imag, label="imag, difference, oneband")
    plt.plot(siwk_dga_2.real - siwk_test_2[niv_test - 40 : niv_test + 40].real, label="real, difference, twoband")
    plt.plot(siwk_dga_2.imag - siwk_test_2[niv_test - 40 : niv_test + 40].imag, label="imag, difference, twoband")
    plt.legend()
    plt.show()

    import brillouin_zone as bz

    k_grid = bz.KGrid((16, 16, 1), bz.two_dimensional_square_symmetries())
    ham = Hamiltonian()
    ham, k_points = ham.read_hk_w2k(
        "/home/julpe/Documents/DATA/Multiorb-DATA/oneband_as_twoband_diagonal_higher_stat_for_vertex_2/wannier.hk"
    )
    ham = ham.set_ek(ham.get_ek(k_grid).reshape(16, 16, 1, 2, 2))
    ek_2 = ham.get_ek(k_grid)

    ham = ham.read_hr_w2k("/home/julpe/Documents/DATA/Singleorb-DATA/N085/wannier_hr.dat")
    ek_1 = ham.get_ek(k_grid)

    assert np.allclose(
        ek_1[..., 0, 0], ek_2[..., 0, 0]
    ), "EK values do not match between oneband and twoband Hamiltonians."

    assert np.allclose(
        ek_1[..., 0, 0], ek_2[..., 1, 1]
    ), "EK values do not match between oneband and twoband Hamiltonians."

    assert np.allclose(ek_2[..., 1, 0], 0)
    assert np.allclose(ek_2[..., 0, 1], 0)

    kx = 64
    k_grid = bz.KGrid((kx, kx, 1), bz.two_dimensional_square_symmetries())
    ham = Hamiltonian()
    ham = ham.read_hr_w2k("/home/julpe/Documents/DATA/Singleorb-DATA/N085/wannier_hr.dat")
    ek_1 = ham.get_ek(k_grid)
    ek_stacked = np.zeros((kx, kx, 1, 2, 2), dtype=ek_1.dtype)
    ek_stacked[..., 0, 0] = ek_1[..., 0, 0]
    ek_stacked[..., 1, 1] = ek_1[..., 0, 0]
    ham.set_ek(ek_stacked)
    ham.write_hk_w2k(
        f"/home/julpe/Documents/DATA/Multiorb-DATA/oneband_as_twoband_diagonal_higher_stat_for_vertex_2/wannier_{kx}x{kx}.hk",
        k_grid,
        ek_stacked,
    )

    g_dmft_1 = np.load(os.path.join(oneband_folder, "g_dmft.npy"))[0, 0, niv_dmft - 100 : niv_dmft + 100]
    g_dmft_2 = np.load(os.path.join(twoband_folder, "g_dmft.npy"))[0, 0, niv_dmft - 100 : niv_dmft + 100]

    plt.figure()
    plt.plot(g_dmft_1.real - g_dmft_2.real, label="real, difference")
    plt.plot(g_dmft_1.imag - g_dmft_2.imag, label="imag, difference")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(g_loc_1.real - g_dmft_1.real, label="real, g_loc - g_dmft, oneband")
    plt.plot(g_loc_1.imag - g_dmft_1.imag, label="imag, g_loc - g_dmft, oneband")
    plt.plot(g_loc_2.real - g_dmft_2.real, label="real, g_loc - g_dmft, twoband")
    plt.plot(g_loc_2.imag - g_dmft_2.imag, label="imag, g_loc - g_dmft, twoband")
    plt.legend()
    plt.show()

    gchi0_test_1 = -12.5 * g_loc_1 * g_loc_1
    gchi0_test_2 = -12.5 * g_loc_2 * g_loc_2
    gchi0_dmft_test_1 = -12.5 * g_dmft_1 * g_dmft_1
    gchi0_dmft_test_2 = -12.5 * g_dmft_2 * g_dmft_2

    gchi0_q_test_1 = (-12.5 * g_dga_1_nonlocal * g_dga_1_nonlocal)[..., niv_dmft - 100 : niv_dmft + 100].sum(
        axis=(0, 1, 2)
    )
    gchi0_q_test_2 = (-12.5 * g_dga_2_nonlocal * g_dga_2_nonlocal)[..., niv_dmft - 100 : niv_dmft + 100].sum(
        axis=(0, 1, 2)
    )

    plt.figure()
    plt.plot(gchi0_test_1.real - gchi0_test_2.real, label="real, gchi0_test")
    plt.plot(gchi0_test_1.imag - gchi0_test_2.imag, label="imag, gchi0_test")
    plt.plot(gchi0_dmft_test_1.real - gchi0_dmft_test_2.real, label="real, gchi0_dmft_test")
    plt.plot(gchi0_dmft_test_1.imag - gchi0_dmft_test_2.imag, label="imag, gchi0_dmft_test")
    plt.plot(gchi0_q_test_1.real - gchi0_q_test_2.real, label="real, gchi0_q_test")
    plt.plot(gchi0_q_test_1.imag - gchi0_q_test_2.imag, label="imag, gchi0_q_test")
    plt.legend()
    plt.show()

    kernel_1 = np.load(os.path.join(oneband_folder, "kernel.npy"))[:, 0, 0, 0, 0, 0]
    kernel_2 = np.load(os.path.join(twoband_folder, "kernel.npy"))[:, 0, 0, 0, 0, 0]

    plt.figure()
    plt.plot(np.sum(kernel_1.real - kernel_2.real, axis=0), label="kernel, real, difference")
    plt.plot(np.sum(kernel_1.imag - kernel_2.imag, axis=0), label="kernel, imag, difference")
    plt.plot()
    plt.legend()
    plt.show()

    kernel_dc_rank0_1 = np.load(os.path.join(oneband_folder, "kernel_dc_rank_0.npy"))[:, 0, 0, 0, 0, 0]
    kernel_dc_rank0_2 = np.load(os.path.join(twoband_folder, "kernel_dc_rank_0.npy"))[:, 0, 0, 0, 0, 0]

    plt.figure()
    plt.plot(np.max(kernel_dc_rank0_1.real - kernel_dc_rank0_2.real, axis=0), label="real, difference, kernel_dc")
    plt.plot(np.max(kernel_dc_rank0_1.imag - kernel_dc_rank0_2.imag, axis=0), label="imag, difference, kernel_dc")
    plt.plot()
    plt.legend()
    plt.show()

    kernel_dc_rank0_1_00 = np.load(os.path.join(oneband_folder, "kernel_dc_rank_0.npy"))[:, 0, 0, 0, 0, 0]

    kernel_dc_rank0_2_00 = np.load(os.path.join(twoband_folder, "kernel_dc_rank_0.npy"))[:, 0, 0, 0, 0, 0]
    kernel_dc_rank0_2_11 = np.load(os.path.join(twoband_folder, "kernel_dc_rank_0.npy"))[:, 1, 1, 1, 1, 0]

    plt.figure()
    plt.plot(np.sum(kernel_dc_rank0_1_00.real - kernel_dc_rank0_2_00.real, axis=0), label="real, diff dc oneb w twob")
    plt.plot(np.sum(kernel_dc_rank0_1_00.imag - kernel_dc_rank0_2_11.imag, axis=0), label="imag, diff dc oneb w twob")
    plt.plot(np.sum(kernel_dc_rank0_2_00.real - kernel_dc_rank0_2_11.real, axis=0), label="real, diff dc twob w twob")
    plt.plot(np.sum(kernel_dc_rank0_2_00.imag - kernel_dc_rank0_2_11.imag, axis=0), label="imag, diff dc twob w twob")
    plt.plot()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(siw_dmft_1.real - siw_dmft_2.real, label="real, difference, sigma_dmft")
    plt.plot(siw_dmft_1.imag - siw_dmft_2.imag, label="imag, difference, sigma_dmft")
    plt.plot()
    plt.legend()
    plt.show()
