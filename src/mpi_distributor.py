import os

import h5py
import mpi4py.MPI as MPI
import numpy as np

import config


class MpiDistributor:
    """
    Distributes tasks among all available cores. Uses the first (q) dimension to slice the vertex data into chunks
    #and sends it to all active MPI processes.
    """

    def __init__(self, ntasks: int = 1, comm: MPI.Comm = None, name: str = ""):
        self._comm = comm
        self._ntasks = ntasks
        self._file = None
        self._my_slice = None
        self._sizes = None
        self._my_size = None
        self._slices = None

        self._distribute_tasks()

        if config.output.output_path is not None:
            # creates rank file if it does not exist
            self._fname = os.path.join(config.output.output_path, f"{name}_Rank{self.my_rank:05d}.hdf5")
            self._file = h5py.File(self._fname, "a")
            self._file.close()

    def __del__(self):
        if self._file is not None:
            try:
                self.close_file()
            except:
                pass

    def __enter__(self):
        self.open_file()
        return self._file

    def __exit__(self, exc_type, exc_value, traceback):
        if self._file:
            self.close_file()

    @property
    def comm(self) -> MPI.Comm:
        return self._comm

    @property
    def is_root(self) -> bool:
        return self.my_rank == 0

    @property
    def ntasks(self) -> int:
        return self._ntasks

    @property
    def sizes(self) -> np.ndarray:
        return self._sizes

    @property
    def my_rank(self) -> int:
        return self._comm.Get_rank()

    @property
    def my_tasks(self) -> np.ndarray:
        return np.arange(0, self.ntasks)[self.my_slice]

    @property
    def mpi_size(self) -> int:
        return self._comm.size

    @property
    def my_size(self) -> int:
        return self._my_size

    @property
    def my_slice(self) -> int:
        return self._my_slice

    def open_file(self):
        try:
            self._file = h5py.File(self._fname, "r+")
        except:
            pass

    def close_file(self):
        try:
            self._file.close()
        except:
            pass

    def delete_file(self):
        try:
            os.remove(self._fname)
        except:
            pass

    def allgather(self, rank_result: np.ndarray = None) -> np.ndarray:
        tot_shape = (self.ntasks,) + rank_result.shape[1:]
        tot_result = np.empty(tot_shape, rank_result.dtype)
        # tot_result[...] = np.nan
        other_dims = np.prod(rank_result.shape[1:])

        # The sizes argument needs the total number of elements rather than
        # just the first axis. The type argument is inferred.
        self.comm.Allgatherv(rank_result, [tot_result, self.sizes * other_dims])
        return tot_result

    def gather(self, rank_result: np.ndarray = None, root: int = 0) -> np.ndarray:
        """Gather numpy array from ranks."""
        tot_shape = (self.ntasks,) + rank_result.shape[1:]
        tot_result = np.empty(tot_shape, rank_result.dtype)
        other_dims = np.prod(rank_result.shape[1:])
        self.comm.Gatherv(rank_result, [tot_result, self.sizes * other_dims], root=root)
        return tot_result

    def scatter(self, full_data: np.ndarray = None, root: int = 0) -> np.ndarray:
        """Scatter full_data among ranks using the first dimension."""
        if full_data is not None:
            assert isinstance(full_data, np.ndarray), "full_data must be a numpy array"
            rest_shape = np.shape(full_data)[1:]
            data_type = full_data.dtype
        else:
            rest_shape = None
            data_type = None

        data_type = self.comm.bcast(data_type, root)
        rest_shape = self.comm.bcast(rest_shape, root)
        rank_shape = (self.my_size,) + rest_shape
        rank_data = np.empty(rank_shape, dtype=data_type)
        other_dims = np.prod(rank_data.shape[1:])
        self.comm.Scatterv([full_data, self.sizes * other_dims], rank_data, root=root)
        return rank_data

    def bcast(self, data, root=0):
        """Broadcast data to all ranks."""
        return self.comm.bcast(data, root=root)

    def allreduce(self, rank_result=None) -> np.ndarray:
        tot_result = np.zeros(np.shape(rank_result), dtype=rank_result.dtype)
        self.comm.Allreduce(rank_result, tot_result)
        return tot_result

    @staticmethod
    def create_distributor(ntasks: int, comm: MPI.Comm, name: str = "") -> "MpiDistributor":
        if comm is None:
            comm = MPI.COMM_WORLD
        return MpiDistributor(ntasks=ntasks, comm=comm, name=name)

    def _distribute_tasks(self):
        n_per_rank = self.ntasks // self.mpi_size
        n_excess = self.ntasks - n_per_rank * self.mpi_size
        self._sizes = n_per_rank * np.ones(self.mpi_size, int)

        if n_excess:
            self._sizes[-n_excess:] += 1

        slice_ends = self._sizes.cumsum()
        self._slices = list(map(slice, slice_ends - self._sizes, slice_ends))
        self._my_size = self._sizes[self.my_rank]
        self._my_slice = self._slices[self.my_rank]
