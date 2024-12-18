import gc

from i_have_mat import IHaveMat


class MemoryHelper:
    @staticmethod
    def delete(*args):
        for arg in args:
            if isinstance(arg, IHaveMat):
                del arg.mat
            del arg
        gc.collect()
