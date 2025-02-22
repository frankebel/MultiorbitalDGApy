import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    g_1 = np.load("/home/julpe/Desktop/g_1.npy")
    mat_grid = np.load("/home/julpe/Desktop/mat_grid.npy")

    gchi_dens_paul = np.load("/home/julpe/Desktop/gchi_dens_paul.npy")
    gchi_dens_me = np.load("/home/julpe/Desktop/gchi_dens_me.npy")
    gchi_dens_me = gchi_dens_me[0, 0, 0, 0]

    gchi_magn_paul = np.load("/home/julpe/Desktop/gchi_magn_paul.npy")
    gchi_magn_me = np.load("/home/julpe/Desktop/gchi_magn_me.npy")
    gchi_magn_me = gchi_magn_me[0, 0, 0, 0]

    niv = gchi_magn_me.shape[-1] // 2

    assert np.allclose(gchi_dens_paul, gchi_dens_me, atol=1e-6)
    assert np.allclose(gchi_magn_paul, gchi_magn_me, atol=1e-6)
    assert np.allclose(g_1, mat_grid, atol=1e-6)

    siw_paul = np.load(
        "/home/julpe/Documents/DATA/Singleorb-DATA/N085/LDGA_spch_Nk256_Nq256_wc140_vc140_vs0_9/siw_sde_full.npy"
    )[niv:]
    siw_me = np.load(
        "/home/julpe/Documents/DATA/Singleorb-DATA/N085/LDGA_Nk256_Nq256_wc140_vc140_vs0_93/siw_sde_full.npy"
    )[0, 0, niv:]
    siw_dmft = np.load(
        "/home/julpe/Documents/DATA/Singleorb-DATA/N085/LDGA_Nk256_Nq256_wc140_vc140_vs0_93/sigma_dmft.npy"
    )[0, 0]
    niv_dmft = siw_dmft.shape[-1] // 2
    siw_dmft = siw_dmft[niv_dmft : niv_dmft + niv]

    plt.figure()
    plt.plot(siw_paul.real, label="Paul, real")
    plt.plot(siw_me.real, label="Me, real")
    plt.plot(siw_paul.imag, label="Paul, imag")
    plt.plot(siw_me.imag, label="Me, imag")
    plt.plot(siw_dmft.real, label="DMFT, real")
    plt.plot(siw_dmft.imag, label="DMFT, imag")
    plt.grid()
    plt.legend()
    plt.title("Comparison of the self-energy")
    plt.xlabel(r"$\nu_n$")
    plt.ylabel(r"$\Sigma(i\nu_n)$")
    plt.tight_layout()
    plt.show()

    print("Hellow")
