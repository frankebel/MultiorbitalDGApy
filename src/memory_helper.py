import gc


class MemoryHelper:
    @staticmethod
    def delete(*args):
        for arg in args:
            del arg
        gc.collect()
