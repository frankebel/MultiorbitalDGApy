import itertools
import os
import time

import matplotlib.pyplot as plt
import numpy as np

folder = (
    "/home/julpe/Documents/DATA/Singleorb-DATA/N085_B12_5_low_iter_lower_statistics/LDGA_Nk256_Nq256_wc60_vc40_vs20/"
)
iteration = "1"

filename_oneband = f"/home/julpe/Documents/DATA/Singleorb-DATA/N085/LDGA_Nk1024_Nq1024_wc70_vc40_vs0/sigma_dga.npy"
filename_twoband = f"/home/julpe/Documents/DATA/Multiorb-DATA/oneband_as_twoband_diagonal_higher_stat_for_vertex_2/LDGA_Nk1024_Nq1024_wc60_vc70_vs150/sigma_dga.npy"


def show_self_energy_convergence():
    filename_oneband = "/home/julpe/Documents/DATA/Singleorb-DATA/N085/LDGA_Nk{}_Nq{}_wc60_vc40_vs0/sigma_dga.npy"
    filename_twoband = "/home/julpe/Documents/DATA/Multiorb-DATA/oneband_as_twoband_diagonal_higher_stat_for_vertex_2/LDGA_Nk{}_Nq{}_wc60_vc40_vs0/sigma_dga.npy"

    sums_real = []
    sums_imag = []
    kx_values = [16, 20, 24, 28, 32, 36, 40, 48, 56, 64]

    for kx in kx_values:
        siw_oneband = np.load(filename_oneband.format(kx * kx, kx * kx))[..., 0, 0, :]
        siw_twoband = np.load(filename_twoband.format(kx * kx, kx * kx))[..., 0, 0, :]

        niv = siw_oneband.shape[-1] // 2

        siw_oneband = siw_oneband[..., niv - 40 : niv + 40]
        siw_twoband = siw_twoband[..., niv - 40 : niv + 40]

        sums_real.append(np.sum(np.abs(siw_oneband.real - siw_twoband.real)) / (kx * kx))
        sums_imag.append(np.sum(np.abs(siw_oneband.imag - siw_twoband.imag)) / (kx * kx))

    plt.figure()
    plt.plot(sums_real, label="Real part difference")
    plt.plot(sums_imag, label="Imaginary part difference")
    plt.xlabel("kx (and ky)")
    plt.xticks(ticks=[i for i in range(len(kx_values))], labels=kx_values)
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_self_energies():
    filename_oneband = "/home/julpe/Documents/DATA/Singleorb-DATA/N085/LDGA_Nk{}_Nq{}_wc60_vc40_vs0/sigma_dga.npy"
    filename_twoband = "/home/julpe/Documents/DATA/Multiorb-DATA/oneband_as_twoband_diagonal_higher_stat_for_vertex_2/LDGA_Nk{}_Nq{}_wc60_vc40_vs0/sigma_dga.npy"

    kx_values = [16, 20, 24, 28, 32, 36, 40, 48, 56, 64]

    for kx in kx_values:
        siw_oneband = np.load(filename_oneband.format(kx * kx, kx * kx))[..., 0, 0, :]
        siw_twoband = np.load(filename_twoband.format(kx * kx, kx * kx))[..., 0, 0, :]

        niv = siw_oneband.shape[-1] // 2

        siw_oneband = siw_oneband[..., niv : niv + 40]
        siw_twoband = siw_twoband[..., niv : niv + 40]

        plt.figure()
        plt.plot(np.min(siw_oneband.real, axis=(0, 1, 2)), label="Real, oneband")
        plt.plot(np.min(siw_oneband.imag, axis=(0, 1, 2)), label="Imag, oneband")
        plt.plot(np.min(siw_twoband.real, axis=(0, 1, 2)), label="Real, twoband")
        plt.plot(np.min(siw_twoband.imag, axis=(0, 1, 2)), label="Imag, twoband")
        plt.xlabel(r"$\nu_n$")
        plt.ylabel(r"$\Sigma(\nu_n)$")
        # plt.xticks(ticks=[i for i in range(len(kx_values))], labels=kx_values)
        plt.legend()
        plt.tight_layout()
        plt.show()


def show_self_energy_kx_ky(kx: int, ky: int):
    siw_oneband = np.load(filename_oneband)
    siw_twoband = np.load(filename_twoband)

    niv = siw_oneband.shape[-1] // 2

    siw_oneband = np.reshape(siw_oneband, (16, 16, 1) + siw_oneband.shape[1:])
    siw_oneband = siw_oneband[kx, ky, 0, 0, 0, niv : niv + 80]
    siw_twoband = np.reshape(siw_twoband, (16, 16, 1) + siw_twoband.shape[1:])
    siw_twoband = siw_twoband[kx, ky, 0, 0, 0, niv : niv + 80]

    plt.figure()
    plt.plot(siw_oneband.real, label="real, oneband")
    plt.plot(siw_oneband.imag, label="imag, oneband")
    plt.plot(siw_twoband.real, label="real, twoband")
    plt.plot(siw_twoband.imag, label="imag, twoband")
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()


def show_mean_self_energy(save: bool, path: str):
    siw_oneband = np.load(filename_oneband)
    siw_twoband = np.load(filename_twoband)

    niv = siw_oneband.shape[-1] // 2

    siw_oneband = np.mean(siw_oneband[..., niv : niv + 80], axis=(0, 1, 2))[0, 0]
    siw_twoband = np.mean(siw_twoband[..., niv : niv + 80], axis=(0, 1, 2))[0, 0]

    plt.figure()
    plt.plot(siw_oneband.real, label="real, 1-band")
    plt.plot(siw_oneband.imag, label="imag, 1-band")
    plt.plot(siw_twoband.real, label="real, 2-band")
    plt.plot(siw_twoband.imag, label="imag, 2-band")
    # plt.plot(siw_oneband.real - siw_twoband.real, label="real, difference")
    # plt.plot(siw_oneband.imag - siw_twoband.imag, label="imag, difference")
    plt.xlabel(r"$\nu_n$")
    plt.ylabel(r"$\Sigma(\nu_n)$")
    plt.tight_layout()
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(f"{path}/self_energy_{iteration}.png")
    else:
        plt.show()
    plt.close()


def show_self_energy_2d():
    siw_oneband = np.load(filename_oneband)
    siw_twoband = np.load(filename_twoband)

    niv = siw_oneband.shape[-1] // 2
    # siw_oneband = np.reshape(siw_oneband, (16, 16, 1) + siw_oneband.shape[1:])
    siw_oneband = siw_oneband[..., 0, 0, 0, niv]

    # siw_twoband = np.reshape(siw_twoband, (16, 16, 1) + siw_twoband.shape[1:])
    siw_twoband = siw_twoband[..., 0, 0, 0, niv]

    fig, axes = plt.subplots(2, 3, figsize=(10, 10))  # 2x3 grid

    # First column: with -np.min()
    im1 = axes[0, 0].matshow(siw_oneband.real, cmap="viridis")
    axes[0, 0].set_title("Real, twoband with all gamma entries")

    im2 = axes[1, 0].matshow(siw_oneband.imag, cmap="viridis")
    axes[1, 0].set_title("Imag, twoband with all gamma entries")

    # Second row: with +np.min()
    im3 = axes[0, 1].matshow(siw_twoband.real, cmap="viridis")
    axes[0, 1].set_title("Real, twoband with only some gamma entries")

    im4 = axes[1, 1].matshow(siw_twoband.imag, cmap="viridis")
    axes[1, 1].set_title("Imag, twoband with only some gamma entries")

    im5 = axes[0, 2].matshow(np.abs(siw_oneband - siw_twoband).real, cmap="viridis")
    axes[0, 2].set_title("Real, Difference")
    im6 = axes[1, 2].matshow(np.abs(siw_oneband - siw_twoband).imag, cmap="viridis")
    axes[1, 2].set_title("Imag, Difference")

    fig.colorbar(im1, ax=axes[0, 0], aspect=15, fraction=0.08, location="right", pad=0.05)
    fig.colorbar(im2, ax=axes[1, 0], aspect=15, fraction=0.08, location="right", pad=0.05)
    fig.colorbar(im3, ax=axes[0, 1], aspect=15, fraction=0.08, location="right", pad=0.05)
    fig.colorbar(im4, ax=axes[1, 1], aspect=15, fraction=0.08, location="right", pad=0.05)
    fig.colorbar(im5, ax=axes[0, 2], aspect=15, fraction=0.08, location="right", pad=0.05)
    fig.colorbar(im6, ax=axes[1, 2], aspect=15, fraction=0.08, location="right", pad=0.05)

    # Adjust layout
    fig.tight_layout()
    fig.legend()
    plt.grid()
    fig.show()


def show_mu_history():
    mu_history = np.load(os.path.join(folder, "mu_history.npy"))

    plt.figure()
    plt.plot(mu_history)
    plt.xlabel("Iteration")
    plt.ylabel(r"$\mu$")
    plt.title(r"$\mu$ history")
    plt.grid()
    plt.show()


def component2index_general(num_bands: int, bands: list, spins: list) -> int:
    assert num_bands > 0, "Number of bands has to be set to non-zero positive integers."

    n_spins = 2
    dims_bs = 4 * (num_bands * n_spins,)
    dims_1 = (num_bands, n_spins)

    bandspin = np.ravel_multi_index((bands, spins), dims_1)
    return np.ravel_multi_index(bandspin, dims_bs) + 1


def get_worm_components(num_bands: int) -> list[int]:
    orbs = [list(orb) for orb in itertools.product(range(num_bands), repeat=4)]
    spins = [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 1, 0]
    component_indices = []
    for o in orbs:
        for s in spins:
            component_indices.append(int(component2index_general(num_bands, o, s)))
    return sorted(component_indices)


if __name__ == "__main__":
    # indices = get_worm_components(num_bands=3)
    # print(indices)
    # print(len(indices))

    # show_self_energy_convergence()
    # show_self_energies()

    """
    filename_oneband = "/home/julpe/Documents/DATA/Multiorb-DATA/oneband_as_twoband_diagonal_higher_stat_for_vertex_2/LDGA_Nk256_Nq256_wc30_vc20_vs0_all_gamma/sigma_dga.npy"
    filename_twoband = "/home/julpe/Documents/DATA/Multiorb-DATA/oneband_as_twoband_diagonal_higher_stat_for_vertex_2/LDGA_Nk256_Nq256_wc30_vc20_vs0_some_gamma_{}/sigma_dga.npy"

    for i in range(1, 4):
        filename_twoband = filename_twoband.format(i)

        show_self_energy_2d()

        siw_1 = np.load(filename_oneband)[..., 0, 0, 1000 : 1000 + 40]
        siw_2 = np.load(filename_twoband)[..., 0, 0, 1000 : 1000 + 40]

        siw_1 = np.max(siw_1, axis=(0, 1, 2))
        siw_2 = np.max(siw_2, axis=(0, 1, 2))

        plt.figure()

        plt.figure()
        plt.plot(siw_1.real, label="real, all gamma")
        plt.plot(siw_1.imag, label="imag, all gamma")
        plt.plot(siw_2.real, label="real, some gamma")
        plt.plot(siw_2.imag, label="imag, some gamma")
        plt.tight_layout()
        plt.legend()
        plt.show()
    """

    for kx in [16, 20, 24, 28, 32, 36, 40, 48, 56, 64]:
        filename_oneband = "/home/julpe/Documents/DATA/Multiorb-DATA/oneband_as_twoband_diagonal_higher_stat_for_vertex_2/LDGA_Nk{}_Nq{}_wc60_vc70_vs0/sigma_dga.npy"
        filename_twoband = "/home/julpe/Documents/DATA/Singleorb-DATA/N085/LDGA_Nk{}_Nq{}_wc60_vc70_vs0/sigma_dga.npy"

        siw_1 = np.load(filename_oneband.format(kx * kx, kx * kx))[..., 0, 0, 1000 : 1000 + 40]
        siw_2 = np.load(filename_twoband.format(kx * kx, kx * kx))[..., 0, 0, 1000 : 1000 + 40]

        siw_1 = np.mean(siw_1, axis=(0, 1, 2))
        siw_2 = np.mean(siw_2, axis=(0, 1, 2))

        plt.figure()
        plt.plot(siw_1.real, label="real, all gamma")
        plt.plot(siw_1.imag, label="imag, all gamma")
        plt.plot(siw_2.real, label="real, some gamma")
        plt.plot(siw_2.imag, label="imag, some gamma")
        plt.tight_layout()
        plt.legend()
        plt.show()

    # show_mean_self_energy(False, "")
    # show_self_energy_2d()
    # show_self_energy_kx_ky(1, 0)
    # show_mu_history()

    # ek_1 = np.load(
    #    f"/home/julpe/Documents/DATA/Singleorb-DATA/N085_B12_5_low_iter/LDGA_Nk256_Nq256_wc60_vc40_vs20/ek.npy"
    # )
    # ek_2 = np.load(
    #    f"/home/julpe/Documents/DATA/Multiorb-DATA/oneband_as_twoband_diagonal/LDGA_Nk256_Nq256_wc60_vc40_vs20/ek.npy"
    # )
    # diff = np.sum(2 * ek_1[..., 0, 0] - ek_2[..., 0, 0] - ek_2[..., 1, 1])
    # test = np.sum(ek_2[..., 1, 0] + ek_2[..., 0, 1])
    # print(diff, test)
