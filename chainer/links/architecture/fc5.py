from chainer import link
from chainer.links.connection import linear


class FC5(link.ChainList):
    """

    https://github.com/Alexey-Kamenev/Benchmarks/blob/master/CNTK/ffn.config

    """

    def __init__(self):
        super(FC5, self).__init__(linear.Linear(512, 2048),
                                  linear.Linear(2048, 2048),
                                  linear.Linear(2048, 2048),
                                  linear.Linear(2048, 2048),
                                  linear.Linear(2048, 10000))

    def __call__(self, x):
        for l in self:
            x = l(x)
        return x
