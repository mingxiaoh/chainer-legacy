from chainer import link
from chainer.functions.noise import dropout
from chainer.links.connection import lstm
from chainer.links.connection import linear


class BigLSTM(link.Chain):
    """

    http://arxiv.org/abs/1602.02410v2
    
    """

    def __init__(self):
        lstms = link.ChainList(lstm.LSTM(10, 20),
                               lstm.LSTM(20, 20))
        linears = link.ChainList(linear.Linear(20, 10),
                                 linear.Linear(10, 2))
        super(BigLSTM, self).__init__(lstms=lstms, linears=linears)
        self.train = True
        # initialize the bias vector of forget gates by 1.0.

    def reset_state(self):
        for l in self.lstms:
            l.reset_state()

    def __call__(self, xs):
        ret = []
        for x in xs:
            for l in self.lstms:
                x = l(x)
                x = dropout.dropout(x, 0.25, self.train)
            for l in self.linears:
                x = l(x)
            ret.append(x)
        return ret
