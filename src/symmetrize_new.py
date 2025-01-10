import itertools as it

import h5py
import numpy as np


def index2component_general(num_bands: int, n: int, ind: int):
    bandspin = np.zeros(n, dtype=np.int_)
    spin = np.zeros(n, dtype=np.int_)
    band = np.zeros(n, dtype=np.int_)
    ind_tmp = ind - 1
    tmp = (2 * num_bands) ** np.arange(n, -1, -1)

    for i in range(n):
        bandspin[i] = ind_tmp // tmp[i + 1]
        spin[i] = bandspin[i] % 2
        band[i] = bandspin[i] // 2
        ind_tmp -= tmp[i + 1] * bandspin[i]

    return bandspin, band, spin


# compute a compound index from orbital indices only.
def component2index_band(num_bands: int, n: int, b: list):
    return 1 + sum(num_bands ** (n - i - 1) * b[i] for i in range(n))


# compute the orbital indices from a compound index
def index2component_band(num_bands, n, ind):
    return [(ind - 1) // (num_bands ** (n - i - 1)) % num_bands for i in range(n)]


if __name__ == "__main__":
    # input_filename = input("Enter the filename of the HDF5 vertex file: ")
    # output_filename = input("Enter the output filename: ")

    input_filename = "/home/julpe/Documents/DATA/Multiorb-DATA/Vertex.hdf5"
    output_filename = "/home/julpe/Documents/DATA/Multiorb-DATA/g4iw_sym.hdf5"

    file = h5py.File(input_filename, "r")

    n_bands = int(file[".config"].attrs[f"atoms.1.nd"]) + int(file[".config"].attrs[f"atoms.1.np"])

    g4iw_groupstring = "worm-last/ineq-001/g4iw-worm"
    indices = list(file[g4iw_groupstring].keys())

    print(f"Nonzero number of elements of g2 in dataset: {len(indices)} / {(2*n_bands)**4}")

    # determination of niw and niv
    first_element_shape = file[f"{g4iw_groupstring}/{indices[0]}/value"].shape
    assert first_element_shape[0] % 2 == 0
    assert first_element_shape[-1] % 2 != 0
    niv = first_element_shape[0] // 2
    niw = first_element_shape[-1] // 2

    print("Number of bands:", n_bands)
    print("Number of fermionic Matsubara frequencies:", niv)
    print("Number of bosonic Matsubara frequencies:", niw)

    elements = np.array([file[f"{g4iw_groupstring}/{idx}/value"] for idx in indices])
    elements = elements.transpose(0, -1, 1, 2)

    # for some reason, the elements are stored transposed in vv' in symmetrize_old.py
    # therefore, every time we have to read or write, we have to transpose in vv' (do not confuse with the transpose above)
    elements = elements.transpose(0, 1, 3, 2)

    # construct G2dens and G2magn the output file
    bands, spins = zip(*(index2component_general(n_bands, 4, int(i))[1:3] for i in indices))

    # since we are SU(2) symmetric, we only have to pick out the elements, where spin is either
    # [0,0,0,0] or [1,1,1,1] for uu component and [0,0,1,1] or [1,1,0,0] for dd component
    # that means for each band, we only have to pick two spin combinations
    g2_dens = np.zeros((n_bands, n_bands, n_bands, n_bands, 2 * niw + 1, 2 * niv, 2 * niv), dtype=np.complex128)
    g2_magn = np.zeros((n_bands, n_bands, n_bands, n_bands, 2 * niw + 1, 2 * niv, 2 * niv), dtype=np.complex128)
    spin_uu, spin_ud = [0, 0, 0, 0], [0, 0, 1, 1]

    for i, j, k, l in it.product(range(n_bands), repeat=4):
        target_orbital = [i, j, k, l]
        print(f"Calculating G2 (dens & magn) for orbitals {target_orbital} ...")

        idx_uu, idx_ud = (
            next(
                (
                    idx
                    for idx, band in enumerate(bands)
                    if np.array_equal(band, target_orbital) and np.array_equal(spins[idx], spin)
                ),
                None,
            )
            for spin in (spin_uu, spin_ud)
        )

        if idx_uu is None or idx_ud is None:
            continue

        g2_dens[i, j, k, l] = elements[idx_uu] + elements[idx_ud]
        g2_magn[i, j, k, l] = elements[idx_uu] - elements[idx_ud]

    g2_dens[np.isnan(g2_dens)] = 0.0 + 1j * 0.0
    g2_magn[np.isnan(g2_magn)] = 0.0 + 1j * 0.0

    file.close()

    with h5py.File(output_filename, "w") as outfile:
        print("Writing to file ... ")
        for wn in range(2 * niw + 1):
            for i, j, k, l in it.product(range(n_bands), repeat=4):
                compound_index = component2index_band(n_bands, 4, [i, j, k, l])
                outfile[f"ineq-001/dens/{wn:05}/{compound_index:05}/value"] = g2_dens[i, j, k, l, wn].transpose()
                outfile[f"ineq-001/magn/{wn:05}/{compound_index:05}/value"] = g2_magn[i, j, k, l, wn].transpose()

    print("Done!")
