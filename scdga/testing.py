import itertools
import os

import matplotlib.pyplot as plt
import numpy as np

folder = (
    "/home/julpe/Documents/DATA/Singleorb-DATA/N085_B12_5_low_iter_lower_statistics/LDGA_Nk256_Nq256_wc60_vc40_vs20/"
)
iteration = "1"

filename_oneband = f"/home/julpe/Documents/DATA/Singleorb-DATA/N085_B12_5_low_iter_lower_statistics/LDGA_Nk256_Nq256_wc60_vc40_vs20/sigma_dga_iteration_1.npy"
filename_twoband = f"/home/julpe/Documents/DATA/Multiorb-DATA/oneband_as_twoband_diagonal/LDGA_Nk256_Nq256_wc60_vc40_vs20/sigma_dga_iteration_1.npy"


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

    siw_oneband = np.mean(siw_oneband[..., niv : niv + 80], axis=0)[0, 0]
    siw_twoband = np.mean(siw_twoband[..., niv : niv + 80], axis=0)[0, 0]

    plt.figure()
    plt.plot(siw_oneband.real, label="real, 1-band")
    plt.plot(siw_oneband.imag, label="imag, 1-band")
    plt.plot(siw_twoband.real, label="real, 2-band")
    plt.plot(siw_twoband.imag, label="imag, 2-band")
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
    siw_oneband = np.reshape(siw_oneband, (16, 16, 1) + siw_oneband.shape[1:])
    siw_oneband = siw_oneband[..., 0, 0, 0, niv]

    siw_twoband = np.reshape(siw_twoband, (16, 16, 1) + siw_twoband.shape[1:])
    siw_twoband = siw_twoband[..., 0, 0, 0, niv]

    fig, axes = plt.subplots(2, 3, figsize=(10, 10))  # 2x3 grid

    # First column: with -np.min()
    im1 = axes[0, 0].matshow(siw_oneband.real, cmap="viridis")
    axes[0, 0].set_title("Real, oneband")

    im2 = axes[1, 0].matshow(siw_oneband.imag, cmap="viridis")
    axes[1, 0].set_title("Imag, oneband")

    # Second row: with +np.min()
    im3 = axes[0, 1].matshow(siw_twoband.real, cmap="viridis")
    axes[0, 1].set_title("Real, twoband")

    im4 = axes[1, 1].matshow(siw_twoband.imag, cmap="viridis")
    axes[1, 1].set_title("Imag, twoband")

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
    # n_bands = 2
    # indices = get_worm_components(n_bands)
    # print(indices)
    # print(len(indices))

    show_mean_self_energy(False, "")
    # show_self_energy_2d()
    # show_self_energy_kx_ky(4, 0)
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
