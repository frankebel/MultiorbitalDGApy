import matplotlib.pyplot as plt
import numpy as np
import symmetrize_new

if __name__ == "__main__":
    for idx in [1, 4, 7, 10, 13, 16]:
        print(symmetrize_new.index2component_general(1, 4, idx))

    exit()
    siw_dmft = np.load("/home/julpe/Desktop/sigma_dmft.npy", allow_pickle=True)
    siw_mycode = np.load("/home/julpe/Desktop/siw_sde_full.npy", allow_pickle=True)
    siw_emery = np.load("/home/julpe/Desktop/siw_sde_emery.npy", allow_pickle=True)

    niv = siw_emery.shape[-1] // 2

    siw_dmft = siw_dmft[0, 0, siw_dmft.shape[-1] // 2 : siw_dmft.shape[-1] // 2 + niv]
    siw_mycode = siw_mycode[0, 0, niv:]
    siw_emery = siw_emery[niv:]

    plt.figure()
    plt.plot(siw_dmft.real, label="DMFT")
    plt.plot(siw_mycode.real, label="My code")
    plt.plot(siw_emery.real, label="Emery")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("test")
