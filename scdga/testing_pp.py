import matplotlib.pyplot as plt
import numpy as np


def show(obj, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))  # 2x3 grid

    im1 = axes[0].matshow(obj.real, cmap="RdBu")
    axes[0].set_title(f"{title}, Real")

    im2 = axes[1].matshow(obj.imag, cmap="RdBu")
    axes[1].set_title(f"{title}, Imag")

    fig.colorbar(im1, ax=axes[0], aspect=15, fraction=0.08, location="right", pad=0.05)
    fig.colorbar(im2, ax=axes[1], aspect=15, fraction=0.08, location="right", pad=0.05)

    # Adjust layout
    fig.suptitle(title)
    fig.tight_layout()
    fig.legend()
    plt.grid()
    fig.show()


if __name__ == "__main__":
    folder = "/home/julpe/Documents/DATA/Singleorb-DATA/juraj_oneband/LDGA_Nk2304_Nq2304_wc60_vc40_vs20_1/Eliashberg"
    phi_sing_loc_pp = np.load(f"{folder}/phi_sing_loc_pp.npy")[0, 0, 0, 0]
    phi_trip_loc_pp = np.load(f"{folder}/phi_trip_loc_pp.npy")[0, 0, 0, 0]
    phi_ud_loc_pp = np.load(f"{folder}/phi_ud_loc_pp.npy")[0, 0, 0, 0]

    f_ud_loc_pp = np.load(f"{folder}/f_ud_loc_pp.npy")[0, 0, 0, 0, 0]

    # for i in range(f_ud_loc_pp.shape[0]):
    #    show(f_ud_loc_pp[i])

    show(f_ud_loc_pp, "F_ud_loc_pp")
    show(phi_ud_loc_pp, "Phi_ud_loc_pp")
