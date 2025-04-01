import logging
import os
from datetime import datetime

import mpi4py.MPI as MPI


class DgaLogger:
    def __init__(self, comm: MPI.Comm, output_path: str = "./", filename: str = "dga.log"):
        self._comm = comm
        self._output_path = output_path
        self._filename = filename
        self._filepath = os.path.join(output_path, filename)

        if self.is_root:
            self._logger = logging.getLogger("dga_logger")
            self._logger.setLevel(logging.DEBUG)
            self._logger.addHandler(logging.StreamHandler())
            self._logger.info("       Current-T        |    Elapsed-T    |    Message")
            # self._logger.addHandler(logging.FileHandler(self._filepath))

        self._start_time = datetime.now()

    @property
    def is_root(self) -> bool:
        return self._comm.rank == 0

    @property
    def total_elapsed_time(self) -> str:
        delta = datetime.now() - self._start_time
        total_seconds = int(delta.total_seconds())
        days, remainder = divmod(total_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = delta.microseconds // 1000

        return str(f"{days:02}-{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}")

    @property
    def current_time(self) -> str:
        return str(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))[:-3]

    def log(self, message: str, level: int):
        if not self.is_root or message is None or message == "" or level < 0:
            return
        self._logger.log(level, f"{self.current_time} | {self.total_elapsed_time} | {message}")

    def log_debug(self, message: str):
        self.log(" ::DEBUG:: " + message, level=logging.DEBUG)

    def log_info(self, message: str):
        self.log(message, level=logging.INFO)

    def log_memory_usage(self, obj_name: str, memory_usage_in_gb: float, n_exists: int = 1):
        total_object_memory = memory_usage_in_gb * n_exists
        self.log_info(f"{obj_name} {"uses" if n_exists == 1 else "use"} (GB): {total_object_memory:.6f}")
