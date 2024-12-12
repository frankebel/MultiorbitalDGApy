import matplotlib.pyplot as plt
import numpy as np


def cut(mat, niv_core):
    if len(mat.shape) == 3:
        mat = mat[0, 0, ...]
    niv_total = mat.shape[-1] // 2
    return mat[niv_total : niv_total + niv_core]


if __name__ == "__main__":
    siw_my_code: np.ndarray = np.load(
        "/home/julpe/Documents/repos/MultiorbitalDGApy/siw_sde_full.npy", allow_pickle=True
    )

    niv_core = siw_my_code.shape[-1] // 2

    siw_pauls_code_no_asympt: np.ndarray = np.load(
        "/home/julpe/Documents/DATA/LDGA_spch_Nk676_Nq676_wc140_vc140_vs0/siw_sde_full.npy", allow_pickle=True
    )
    siw_pauls_code_with_asympt: np.ndarray = np.load(
        "/home/julpe/Documents/DATA/LDGA_spch_Nk676_Nq676_wc140_vc140_vs100/siw_sde_full.npy", allow_pickle=True
    )
    siw_dmft: np.ndarray = np.load("/home/julpe/Documents/repos/MultiorbitalDGApy/sigma_dmft.npy", allow_pickle=True)

    siw_my_code = cut(siw_my_code, niv_core=niv_core)
    siw_pauls_code_no_asympt = cut(siw_pauls_code_no_asympt, niv_core=niv_core)
    siw_pauls_code_with_asympt = cut(siw_pauls_code_with_asympt, niv_core=niv_core)
    siw_dmft = cut(siw_dmft, niv_core=niv_core)

    plt.figure()
    plt.plot(siw_my_code.real, label="My code")
    plt.plot(siw_pauls_code_no_asympt.real, label="Paul's code no asympt")
    plt.plot(siw_pauls_code_with_asympt.real, label="Paul's code with asympt")
    plt.plot(siw_dmft.real, label="DMFT")
    plt.grid()
    plt.legend()
    plt.show()

    print("success")
