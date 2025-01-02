import gc

from n_point_base import *


class MemoryHelper:
    @staticmethod
    def delete(*args):
        for arg in args:
            if isinstance(arg, IHaveMat):
                del arg.mat
            del arg
        gc.collect()
