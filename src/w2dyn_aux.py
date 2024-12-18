import h5py
import numpy as np

from i_have_channel import Channel


class W2dynFile:
    def __init__(self, fname=None):
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

    def get_nd(self, atom=1):
        return self._from_atom_config("nd", atom=atom)

    def get_np(self, atom=1):
        return self._from_atom_config("np", atom=atom)

    def get_beta(self):
        return self._file[".config"].attrs["general.beta"]

    def get_mu(self, dmft_iter="dmft-last"):
        return self._file[dmft_iter + "/mu/value"][()]

    def get_totdens(self):
        return self._file[".config"].attrs["general.totdens"]

    def get_jdd(self, atom=1):
        return self._from_atom_config("jdd", atom=atom)

    def get_jdp(self, atom=1):
        return self._from_atom_config("jdp", atom=atom)

    def get_jpp(self, atom=1):
        return self._from_atom_config("jpp", atom=atom)

    def get_jppod(self, atom=1):
        """Offdiagonal terms for jpp"""
        return self._from_atom_config("jppod", atom=atom)

    def get_udd(self, atom=1):
        return self._from_atom_config("udd", atom=atom)

    def get_udp(self, atom=1):
        return self._from_atom_config("udp", atom=atom)

    def get_upp(self, atom=1):
        return self._from_atom_config("upp", atom=atom)

    def get_uppod(self, atom=1):
        """Offdiagonal terms for upp"""
        return self._from_atom_config("uppod", atom=atom)

    def get_vdd(self, atom=1):
        return self._from_atom_config("vdd", atom=atom)

    def get_vpp(self, atom=1):
        return self._from_atom_config("vpp", atom=atom)

    def get_siw(self, dmft_iter="dmft-last", atom=1):
        return self._file[self.atom_group(dmft_iter=dmft_iter, atom=atom) + "/siw/value"][()]

    def get_giw(self, dmft_iter="dmft-last", atom=1):
        return self._file[self.atom_group(dmft_iter=dmft_iter, atom=atom) + "/giw/value"][()]

    def get_occ(self, dmft_iter="dmft-last", atom=1):
        return self._file[self.atom_group(dmft_iter=dmft_iter, atom=atom) + "/occ/value"][()]

    def _from_atom_config(self, key, atom=1):
        return self._file[".config"].attrs[f"atoms.{atom:1}.{key}"]


class W2dynG4iwFile:
    def __init__(self, fname: str = None):
        self._fname = fname
        self._file = None
        self.open()

    def __del__(self):
        self._file.close()

    def close(self):
        self._file.close()

    def open(self):
        self._file = h5py.File(self._fname, "r")

    def read_g2_full(self, ineq=1, channel: Channel = Channel.NONE, spinband=1):
        group_exists = True
        g2 = []
        wn = 0
        while group_exists:
            try:
                g2.append(self._file[f"/ineq-{ineq:03}/" + channel.value + f"/{wn:05}/{spinband:05}/value"][()].T)
                wn += 1
            except:
                group_exists = False
        return np.array(g2, dtype=np.complex64)
