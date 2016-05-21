from chainer import link
from chainer.functions.noise import dropout
from chainer.links.connection import lstm
from chainer.links.connection import linear


class SmallLSTM(link.ChainList):
    """

    https://github.com/karpathy/char-rnn/blob/master/train.lua#L38-L48

    """

    def __init__(self):
        super(SmallLSTM, self).__init__(lstm.LSTM(128, 128),
                                        lstm.LSTM(128, 128))

    def reset_state(self):
        for l in self:
            l.reset_state()

    def __call__(self, xs):
        ret = []
        for x in xs:
            for l in self:
                x = l(x)
            ret.append(x)
        return ret
