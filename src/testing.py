import numpy as np


if __name__ == "__main__":
    ggv_mat_working = np.load("/home/julpe/Desktop/Working_code/ggv_mat.npy")[0, 0, 0, 0]
    ggv_mat_testing = np.load("/home/julpe/Desktop/Testing_code/ggv_mat.npy")[0, 0, 0, 0, 140]

    res = np.allclose(ggv_mat_working, ggv_mat_testing)

    gamma_dens_working = np.load("/home/julpe/Desktop/Working_code/Gamma_dens.npy")[0, 0, 0, 0]
    gamma_dens_testing = np.load("/home/julpe/Desktop/Testing_code/Gamma_dens.npy")[0, 0, 0, 0]

    arr = np.abs(gamma_dens_working - gamma_dens_testing)
    res2 = np.sum(np.abs(gamma_dens_working - gamma_dens_testing) > 0)

    print("test")
