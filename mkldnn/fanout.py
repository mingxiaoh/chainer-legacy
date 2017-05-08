class fanout(object):
    current = {}
    def __new__(cls, rank):
        ret = cls.current.get(rank)

        if ret is None:
            ret = 0

        cls.current[rank] = ret + 1
        return ret

    @classmethod
    def clear(cls):
        cls.current = {}
