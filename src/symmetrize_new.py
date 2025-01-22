import gc
import itertools as it
import logging

import h5py
import numpy as np


def index2component_general(num_bands: int, n: int, ind: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


# compute orbital indices from a compound index.
def index2component_band(num_bands: int, n: int, ind: int) -> list:
    b = []
    ind_tmp = ind - 1
    for i in range(n):
        b.append(ind_tmp // (num_bands ** (n - i - 1)))
        ind_tmp = ind_tmp - b[i] * (num_bands ** (n - i - 1))
    return b


# compute a compound index from orbital indices only.
def component2index_band(num_bands: int, n: int, b: list) -> int:
    ind = 1
    for i in range(n):
        ind = ind + num_bands ** (n - i - 1) * b[i]
    return ind


def extract_g2_general(group_string: str, indices: list):
    print(f"Nonzero number of elements of g2 in dataset: {len(indices)} / {(2 * n_bands) ** 4}")

    elements = np.array([vertex_file[f"{group_string}/{idx}/value"] for idx in indices])
    elements = elements.transpose(0, -1, 1, 2)

    # for some reason, the elements are stored transposed in vv' in symmetrize_old.py
    # therefore, every time we have to read or write, we have to transpose in vv' (do not confuse with the transpose above)
    elements = elements.transpose(0, 1, 3, 2)

    # construct G2dens and G2magn the output file
    bands, spins = zip(*(index2component_general(n_bands, 4, int(i))[1:3] for i in indices))

    # since we are SU(2) symmetric, we only have to pick out the elements, where spin is either
    # [0,0,0,0] or [1,1,1,1] for uu component and [0,0,1,1] or [1,1,0,0] for ud component
    # that means for each band, we only have to pick two spin combinations
    g2_uuuu, g2_dddd, g2_dduu, g2_uudd, g2_uddu, g2_duud = (
        np.zeros((n_bands, n_bands, n_bands, n_bands, 2 * niw + 1, 2 * niv, 2 * niv), dtype=np.complex128)
        for _ in range(6)
    )

    spin_dddd, spin_uuuu = [0, 0, 0, 0], [1, 1, 1, 1]
    spin_dduu, spin_uudd = [0, 0, 1, 1], [1, 1, 0, 0]
    spin_uddu, spin_duud = [1, 0, 0, 1], [0, 1, 1, 0]

    for a, b, c, d in it.product(range(n_bands), repeat=4):
        target_orbital = [a, b, c, d]
        print(f"Collecting G2 for orbitals {target_orbital} ...")

        idx_dddd, idx_dduu, idx_uudd, idx_uuuu, idx_uddu, idx_duud = (
            next(
                (
                    idx
                    for idx, band in enumerate(bands)
                    if np.array_equal(band, target_orbital) and np.array_equal(spins[idx], spin)
                ),
                None,
            )
            for spin in (spin_dddd, spin_dduu, spin_uudd, spin_uuuu, spin_uddu, spin_duud)
        )

        if (
            idx_dddd is None
            or idx_dduu is None
            or idx_uudd is None
            or idx_uuuu is None
            or idx_uddu is None
            or idx_duud is None
        ):
            continue

        g2_uuuu[a, b, c, d] = elements[idx_uuuu]
        g2_dddd[a, b, c, d] = elements[idx_dddd]
        g2_dduu[a, b, c, d] = elements[idx_dduu]
        g2_uudd[a, b, c, d] = elements[idx_uudd]
        g2_uddu[a, b, c, d] = elements[idx_uddu]
        g2_duud[a, b, c, d] = elements[idx_duud]

    return g2_uuuu, g2_dddd, g2_dduu, g2_uudd, g2_uddu, g2_duud


if __name__ == "__main__":
    default_input_filename = "Vertex.hdf5"
    default_output_filename_ph = "g4iw_sym.hdf5"

    input_filename = input(f"Enter the DMFT vertex file name (default = {default_input_filename}): ")
    output_filename = input(f"Enter the output filename (default = {default_output_filename_ph}): ")

    input_filename = input_filename if input_filename else default_input_filename
    output_filename = output_filename if output_filename else default_output_filename_ph

    vertex_file = h5py.File(input_filename, "r")
    output_file = h5py.File(output_filename, "w")

    n_bands = int(vertex_file[".config"].attrs[f"atoms.1.nd"]) + int(vertex_file[".config"].attrs[f"atoms.1.np"])

    g4iw_ph_groupstring = "worm-last/ineq-001/g4iw-worm"
    g4iw_pp_groupstring = "worm-last/ineq-001/g4iwpp-worm"
    indices_ph = list(vertex_file[g4iw_ph_groupstring].keys())
    indices_pp = None

    try:
        indices_pp = list(vertex_file[g4iw_pp_groupstring].keys())
    except KeyError:
        logging.getLogger().warning("No g4iwpp-worm group found in the input file. No vertex asymptotics will be used.")

    # determination of niw and niv
    first_element_shape = vertex_file[f"{g4iw_ph_groupstring}/{indices_ph[0]}/value"].shape
    assert first_element_shape[0] % 2 == 0
    assert first_element_shape[-1] % 2 != 0
    niv = first_element_shape[0] // 2
    niw = first_element_shape[-1] // 2

    print("Number of bands:", n_bands)
    print("Number of fermionic Matsubara frequencies:", niv)
    print("Number of bosonic Matsubara frequencies:", niw)

    print("Extracting G2ph ...")
    g2_uuuu_ph, g2_dddd_ph, g2_dduu_ph, g2_uudd_ph, g2_uddu_ph, g2_duud_ph = extract_g2_general(
        g4iw_ph_groupstring, indices_ph
    )
    print("G2ph extracted. Calculating G2_dens and G2_magn for ph.")
    g2_dens_ph = 0.5 * (g2_uuuu_ph + g2_dddd_ph + g2_uudd_ph + g2_dduu_ph)
    g2_magn_ph = 0.5 * (g2_uddu_ph + g2_duud_ph)
    print("G2_dens and G2_magn calculated. Writing to file ...")

    for wn in range(2 * niw + 1):
        for i, j, k, l in it.product(range(n_bands), repeat=4):
            compound_index = component2index_band(n_bands, 4, [i, j, k, l])
            output_file[f"ineq-001/dens/{wn:05}/{compound_index:05}/value"] = g2_dens_ph[i, j, k, l, wn].transpose()
            output_file[f"ineq-001/magn/{wn:05}/{compound_index:05}/value"] = g2_magn_ph[i, j, k, l, wn].transpose()

    del g2_uuuu_ph, g2_dddd_ph, g2_dduu_ph, g2_uudd_ph, g2_uddu_ph, g2_duud_ph, g2_dens_ph, g2_magn_ph
    gc.collect()

    if indices_pp is None:
        output_file.close()
        vertex_file.close()
        print("Done!")
        exit()

    print("Extracting G2pp ...")
    _, _, g2_dduu_pp, g2_uudd_pp, _, _ = extract_g2_general(g4iw_pp_groupstring, indices_pp)
    print("G2pp extracted. Writing G2pp_ud to file ...")
    g2_uudd_pp = 0.5 * (g2_uudd_pp + g2_dduu_pp)
    del g2_dduu_pp
    gc.collect()

    for wn in range(2 * niw + 1):
        for i, j, k, l in it.product(range(n_bands), repeat=4):
            compound_index = component2index_band(n_bands, 4, [i, j, k, l])
            output_file[f"ineq-001/g2pp_ud/{wn:05}/{compound_index:05}/value"] = g2_uudd_pp[i, j, k, l, wn].transpose()

    output_file.close()
    vertex_file.close()
    print("Done!")
