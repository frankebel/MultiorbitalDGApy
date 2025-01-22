import h5py

import symmetrize_new as sym
from n_point_base import *


class W2dynFile:
    def __init__(self, fname: str):
        self._file = None
        self._fname = fname
        self.open()

    def __del__(self):
        self._file.close()

    def close(self):
        self._file.close()

    def open(self):
        self._file = h5py.File(self._fname, "r")

    def atom_group(self, dmft_iter="dmft-last", atom=1):
        return dmft_iter + f"/ineq-{atom:03}"

    def get_nd(self, atom: int = 1) -> int:
        """Returns the number of d orbitals."""
        return self._from_atom_config("nd", atom=atom)

    def get_np(self, atom: int = 1) -> int:
        """Returns the number of p orbitals."""
        return self._from_atom_config("np", atom=atom)

    def get_beta(self) -> float:
        """Returns beta."""
        return self._file[".config"].attrs["general.beta"]

    def get_mu(self, dmft_iter: str = "dmft-last"):
        """Returns the chemical potential."""
        return self._file[dmft_iter + "/mu/value"][()]

    def get_totdens(self) -> float:
        """Returns the total particle density."""
        return self._file[".config"].attrs["general.totdens"]

    def get_jdd(self, atom: int = 1) -> float:
        return self._from_atom_config("jdd", atom=atom)

    def get_jdp(self, atom: int = 1) -> float:
        return self._from_atom_config("jdp", atom=atom)

    def get_jpp(self, atom: int = 1) -> float:
        return self._from_atom_config("jpp", atom=atom)

    def get_jppod(self, atom: int = 1) -> float:
        """Offdiagonal terms for jpp"""
        return self._from_atom_config("jppod", atom=atom)

    def get_udd(self, atom: int = 1) -> float:
        return self._from_atom_config("udd", atom=atom)

    def get_udp(self, atom: int = 1) -> float:
        return self._from_atom_config("udp", atom=atom)

    def get_upp(self, atom: int = 1) -> float:
        return self._from_atom_config("upp", atom=atom)

    def get_uppod(self, atom: int = 1):
        """Offdiagonal terms for upp"""
        return self._from_atom_config("uppod", atom=atom)

    def get_vdd(self, atom: int = 1) -> float:
        return self._from_atom_config("vdd", atom=atom)

    def get_vpp(self, atom: int = 1) -> float:
        return self._from_atom_config("vpp", atom=atom)

    def get_siw(self, dmft_iter: str = "dmft-last", atom: int = 1) -> list:
        """Extracts the DMFT self-energy in Matsubara frequency space as [band, spin, iw]."""
        return self._file[self.atom_group(dmft_iter=dmft_iter, atom=atom) + "/siw/value"][()]

    def get_giw(self, dmft_iter: str = "dmft-last", atom: int = 1) -> list:
        """Extracts the DMFT Green's function in Matsubara frequency space as [band, spin, iw]."""
        return self._file[self.atom_group(dmft_iter=dmft_iter, atom=atom) + "/giw/value"][()]

    def get_occ(self, dmft_iter: str = "dmft-last", atom: int = 1) -> list:
        """Extracts the occupation matrix as [band1, spin1, band2, spin2]."""
        return self._file[self.atom_group(dmft_iter=dmft_iter, atom=atom) + "/occ/value"][()]

    def _from_atom_config(self, key: str, atom: int = 1):
        return self._file[".config"].attrs[f"atoms.{atom:1}.{key}"]


class W2dynG4iwFile:
    def __init__(self, fname: str):
        self._fname = fname
        self._file = None
        self.open()

    def __del__(self):
        self._file.close()

    def close(self):
        self._file.close()

    def open(self):
        self._file = h5py.File(self._fname, "r")

    def read_g2_full_multiband(self, n_bands: int, ineq: int = 1, channel: Channel = Channel.DENS) -> np.ndarray:
        # the next lines determine the size of g2, i.e. niw and niv
        channel_group_string = f"/ineq-{ineq:03}/{channel.value}"
        niw_full = len(self._file[channel_group_string].keys())
        # 00000 is the first element. If it does not exist, there are no bosonic frequenices in the G2 and that would be weird
        first_index = int(next(iter(self._file[f"{channel_group_string}/00000"])))
        niv_full = len(
            self._file[f"{channel_group_string}/00000/{first_index:05}/value"][()]
        )  # extract niv from the size of the array

        g2 = np.zeros((n_bands,) * 4 + (niw_full,) + (niv_full,) * 2, dtype=np.complex128)
        for wn in range(niw_full):
            wn_group_string = f"{channel_group_string}/{wn:05}"
            for ind in self._file[wn_group_string].keys():
                bands = sym.index2component_band(n_bands, 4, int(ind))
                val = self._file[f"{wn_group_string}/{ind}/value"][()].T
                g2[bands[0], bands[1], bands[2], bands[3], wn, ...] = val

        return g2
