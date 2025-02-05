import matplotlib.pyplot as plt
import numpy as np
import time

if __name__ == "__main__":
    test = np.random.rand(25000, 25000)
    start_time = time.time()
    test2 = np.linalg.inv(test)
    end_time = time.time()
    print(f"Execution time: {(end_time - start_time):.6f} seconds")
    exit()

    sde_no_asympt = np.load(
        "/home/julpe/Documents/DATA/Multiorb-DATA/kanamori/LDGA_Nk1600_Nq1600_wc80_vc50_vs0/siw_sde_full.npy",
        allow_pickle=True,
    )
    sde_rpa_asympt_70 = np.load(
        "/home/julpe/Documents/DATA/Multiorb-DATA/kanamori/LDGA_Nk1600_Nq1600_wc80_vc50_vs70_3/siw_sde_full.npy",
        allow_pickle=True,
    )
    sde_my_asympt_20 = np.load(
        "/home/julpe/Documents/DATA/Multiorb-DATA/kanamori/LDGA_Nk1600_Nq1600_wc80_vc50_vs20/siw_sde_full.npy",
        allow_pickle=True,
    )
    sde_my_asympt_70 = np.load(
        "/home/julpe/Documents/DATA/Multiorb-DATA/kanamori/LDGA_Nk1600_Nq1600_wc80_vc50_vs70_1/siw_sde_full.npy",
        allow_pickle=True,
    )

    sde_no_asympt = sde_no_asympt[0, 0]
    sde_rpa_asympt_70 = sde_rpa_asympt_70[0, 0]

    diff_20 = abs(sde_no_asympt.shape[-1] - sde_my_asympt_20.shape[-1])
    sde_my_asympt_20 = sde_my_asympt_20[0, 0, diff_20 // 2 : sde_no_asympt.shape[-1] + diff_20 // 2]

    diff_70 = abs(sde_no_asympt.shape[-1] - sde_my_asympt_70.shape[-1])
    sde_my_asympt_70 = sde_my_asympt_70[0, 0, diff_70 // 2 : sde_no_asympt.shape[-1] + diff_70 // 2]

    plt.plot((sde_no_asympt - sde_rpa_asympt_70).real, label="'RPA' asympt 70 diff real")
    plt.plot((sde_no_asympt - sde_my_asympt_20).real, label="My Asympt 20 diff real")
    plt.plot((sde_no_asympt - sde_my_asympt_70).real, label="My Asympt 70 diff real")
    plt.plot((sde_no_asympt - sde_rpa_asympt_70).imag, label="'RPA' asympt 70 diff imag")
    plt.plot((sde_no_asympt - sde_my_asympt_70).imag, label="My Asympt 20 diff imag")
    plt.plot((sde_no_asympt - sde_my_asympt_70).imag, label="My Asympt 70 diff imag")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    print("done")
