import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    siw_dmft = np.load(
        "/home/julpe/Documents/DATA/Singleorb-DATA/N085/LDGA_Nk1024_Nq1024_wc80_vc50_vs0/sigma_dmft.npy"
    )[0, 0, 0, 0, 0]
    siw_dga_local = np.load(
        "/home/julpe/Documents/DATA/Singleorb-DATA/N085/LDGA_Nk1024_Nq1024_wc80_vc50_vs0/siw_sde_full.npy"
    )
    siw_dga_nonlocal = np.load(
        "/home/julpe/Documents/DATA/Singleorb-DATA/N085/LDGA_Nk1024_Nq1024_wc80_vc50_vs0/sigma_dga.npy"
    )

    dmft_niv = siw_dmft.shape[-1] // 2
    niv_core = siw_dga_nonlocal.shape[-1] // 2

    siw_dmft = siw_dmft[..., dmft_niv : dmft_niv + 100]
    siw_dga_local = siw_dga_local[..., niv_core:]
    siw_dga_nonlocal = np.mean(siw_dga_nonlocal[..., niv_core : niv_core + 100], axis=(0, 1, 2))

    # siw = (siw_dga_nonlocal - 0.5 * (siw_dc_d1 - siw_dc_m1))[0, 0, 0, :]

    plt.figure()
    plt.plot(siw_dga_nonlocal.real, label="real, dga")
    plt.plot(siw_dga_nonlocal.imag, label="imag, dga")
    plt.plot(siw_dmft.real, label="real, dmft")
    plt.plot(siw_dmft.imag, label="imag, dmft")
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()
