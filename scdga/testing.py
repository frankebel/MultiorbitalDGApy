import itertools

import matplotlib.pyplot as plt
import numpy as np

from scdga.symmetrize_new import component2index_band

folder = "/home/julpe/Documents/DATA/Singleorb-DATA/N085/LDGA_Nk256_Nq256_wc140_vc80_vs50_6/"
iteration = "1"


def show_self_energy_kx_ky(kx: int, ky: int):
    siw_dmft = np.load(f"{folder}/sigma_dmft.npy")[0, 0, 0, 0, 0]
    # siw_dga_local = np.load(f"{folder}/siw_sde_full.npy")
    siw_dga_nonlocal = np.load(f"{folder}/sigma_dga_iteration_{iteration}.npy")
    siw_paul = np.load("/home/julpe/Desktop/sigma_paul.npy")
    # siw_dga_nonlocal_fit = np.load(f"{folder}/sigma_dga_fitted.npy")

    niv = siw_dga_nonlocal.shape[-1] // 2

    siw_dmft = siw_dmft[..., niv : niv + 80]
    # siw_dga_local = siw_dga_local[..., niv:]
    siw_dga_nonlocal = np.reshape(siw_dga_nonlocal, (16, 16, 1) + siw_dga_nonlocal.shape[1:])
    siw_dga_nonlocal = siw_dga_nonlocal[kx, ky, 0, 0, 0, niv - 60 : niv + 60]
    # siw_dga_nonlocal_fit = np.mean(siw_dga_nonlocal_fit[..., niv : niv + 50], axis=0)[0, 0]
    niv_paul = siw_paul.shape[-1] // 2
    siw_paul = siw_paul[kx, ky, 0, niv_paul - 60 : niv_paul + 60]

    plt.figure()
    plt.plot(siw_dga_nonlocal.real, label="real, dga")
    plt.plot(siw_dga_nonlocal.imag, label="imag, dga")
    # plt.plot(siw_dga_nonlocal_fit.real, label="real, dga_fit")
    # plt.plot(siw_dga_nonlocal_fit.imag, label="imag, dga_fit")
    plt.plot(siw_paul.real, label="real, paul")
    plt.plot(siw_paul.imag, label="imag, paul")
    # plt.plot(siw_dmft.real, label="real, dmft")
    # plt.plot(siw_dmft.imag, label="imag, dmft")
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()


def show_mean_self_energy(save: bool, path: str):
    siw_dmft = np.load(f"{folder}/sigma_dmft.npy")[0, 0, 0, 0, 0]
    # siw_dga_local = np.load(f"{folder}/siw_sde_full.npy")
    siw_dga_nonlocal = np.load(f"{folder}/sigma_dga_iteration_{iteration}.npy")
    siw_paul = np.load(
        "/home/julpe/Documents/DATA/Singleorb-DATA/N085/LDGA_spch_Nk1024_Nq1024_wc140_vc80_vs50/siwk_dga.npy"
    )
    # siw_dga_nonlocal_fit = np.load(f"{folder}/sigma_dga_fitted.npy")

    niv = siw_dga_nonlocal.shape[-1] // 2

    siw_dmft = siw_dmft[..., niv : niv + 80]
    # siw_dga_local = siw_dga_local[..., niv:]
    siw_dga_nonlocal = np.mean(siw_dga_nonlocal[..., niv : niv + 80], axis=0)[0, 0]  # + 0.2 + 1j * 0.2
    # siw_dga_nonlocal_fit = np.mean(siw_dga_nonlocal_fit[..., niv : niv + 50], axis=0)[0, 0]
    niv_paul = siw_paul.shape[-1] // 2
    siw_paul = np.mean(siw_paul[..., niv_paul : niv_paul + 80], axis=(0, 1, 2))

    plt.figure()
    plt.plot(siw_dga_nonlocal.real, label="real, dga")
    plt.plot(siw_dga_nonlocal.imag, label="imag, dga")
    # plt.plot(siw_dga_nonlocal_fit.real, label="real, dga_fit")
    # plt.plot(siw_dga_nonlocal_fit.imag, label="imag, dga_fit")
    plt.plot(siw_paul.real, label="real, paul")
    plt.plot(siw_paul.imag, label="imag, paul")
    # plt.plot(siw_dmft.real, label="real, dmft")
    # plt.plot(siw_dmft.imag, label="imag, dmft")
    plt.ylim(-2.5, 4)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(f"{path}/self_energy_{iteration}.png")
    else:
        plt.show()
    plt.close()


def show_self_energy_2d():
    siw_dmft = np.load(f"{folder}/sigma_dmft.npy")[0, 0, 0, 0, 0]
    # siw_dga_local = np.load(f"{folder}/siw_sde_full.npy")
    siw_dga_nonlocal = np.load(f"{folder}/sigma_dga_iteration_{iteration}.npy")
    # siw_paul = np.load("/home/julpe/Desktop/sigma_paul.npy")
    # siw_dga_nonlocal_fit = np.load(f"{folder}/sigma_dga_fitted.npy")

    niv = siw_dga_nonlocal.shape[-1] // 2

    siw_dmft = siw_dmft[..., niv : niv + 80]
    # siw_dga_local = siw_dga_local[..., niv:]
    siw_dga_nonlocal = np.reshape(siw_dga_nonlocal, (16, 16, 1) + siw_dga_nonlocal.shape[1:])
    siw_dga_nonlocal = siw_dga_nonlocal[..., 0, 0, 0, niv]
    # siw_dga_nonlocal_fit = np.mean(siw_dga_nonlocal_fit[..., niv : niv + 50], axis=0)[0, 0]
    # niv_paul = siw_paul.shape[-1] // 2
    # siw_paul = siw_paul[..., 0, niv_paul]

    # siw_dga_nonlocal = np.roll(siw_dga_nonlocal, (-2, -1), axis=(0, 1))

    # assert np.allclose(siw_dga_nonlocal, siw_paul, atol=1e-3)

    fig, axes = plt.subplots(2, 3, figsize=(10, 10))  # 2x2 grid

    # First row: siw_dga_nonlocal
    im1 = axes[0, 0].matshow(siw_dga_nonlocal.real, cmap="viridis")
    axes[0, 0].set_title("Real, DGA")

    im2 = axes[0, 1].matshow(siw_dga_nonlocal.imag, cmap="viridis")
    axes[0, 1].set_title("Imag, DGA")

    # Second row: siw_paul
    # im3 = axes[1, 0].matshow(siw_paul.real, cmap="viridis")
    # axes[1, 0].set_title("Real, Paul")

    # im4 = axes[1, 1].matshow(siw_paul.imag, cmap="viridis")
    # axes[1, 1].set_title("Imag, Paul")

    # im5 = axes[0, 2].matshow((siw_dga_nonlocal - siw_paul).real, cmap="viridis")
    # axes[0, 2].set_title("Real, Difference")
    # im6 = axes[1, 2].matshow((siw_dga_nonlocal - siw_paul).imag, cmap="viridis")
    # axes[1, 2].set_title("Imag, Difference")

    fig.colorbar(im1, ax=axes[0, 0])
    fig.colorbar(im2, ax=axes[0, 1])
    # fig.colorbar(im3, ax=axes[1, 0])
    # fig.colorbar(im4, ax=axes[1, 1])
    # fig.colorbar(im5, ax=axes[0, 2])
    # fig.colorbar(im6, ax=axes[1, 2])

    # Adjust layout
    fig.tight_layout()
    fig.legend()
    plt.grid()
    fig.show()


def show_mu_history():
    mu_history = np.load(f"{folder}/mu_history.npy")

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
    return component_indices


if __name__ == "__main__":
    n_bands = 2
    indices = get_worm_components(n_bands)
    print(sorted(indices))
    print(len(indices))
    """
    for i in range(1, 101):
        iteration = i
        show_mean_self_energy(save=True, path="/home/julpe/Desktop/plots")
    """
    # show_mean_self_energy(False, "")
    # show_self_energy_2d()
    # show_self_energy_kx_ky(7, 7)
    # show_mu_history()
