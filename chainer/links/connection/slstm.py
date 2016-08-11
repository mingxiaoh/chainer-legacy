from chainer.functions.activation import lstm
from chainer import link
from chainer.links.connection import linear
from chainer import variable


class StatelessSLSTM(link.Chain):

    """Fully-connected LSTM layer.

    This is a fully-connected LSTM layer as a chain. Unlike the
    :func:`~chainer.functions.lstm` function, which is defined as a stateless
    activation function, this chain holds upward and lateral connections as
    child links.

    It also maintains *states*, including the cell state and the output
    at the previous time step. Therefore, it can be used as a *stateful LSTM*.

    Args:
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of output vectors.

    Attributes:
        upward (chainer.links.Linear): Linear layer of upward connections.
        lateral (chainer.links.Linear): Linear layer of lateral connections.
        c (chainer.Variable): Cell states of LSTM units.
        h (chainer.Variable): Output at the previous timestep.

    """
    def __init__(self, in_size, out_size):
        super(SLSTM, self).__init__(
            upward1=linear.Linear(in_size, 4 * out_size),
            upward2=linear.Linear(in_size, 4 * out_size),
            lateral1=linear.Linear(out_size, 4 * out_size, nobias=True),
            lateral2=linear.Linear(out_size, 4 * out_size, nobias=True),
        )
        self.state_size = out_size


    def _make_input(self, c, h, x, upward, lateral):
        lstm_in = upward(x)
        if h is not None:
            lstm_in += lateral(h)
        if c is None:
            c = variable.Variable(
                self.xp.zeros(
                    (len(x1.data), self.state_size), dtype=x.data.dtype),
                volatile='auto')
        return c, lstm_in


    def __call__(self, c1, h1, c2, h2, x1, x2):
        """Updates the internal state and returns the LSTM outputs.

        Args:
            x (~chainer.Variable): A new batch from the input sequence.

        Returns:
            ~chainer.Variable: Outputs of updated LSTM units.

        """
        c1, lstm_in1 = self._make_input(
            c1, h1, x1, self.upward1, self.lateral1)
        c2, lstm_in2 = self._make_input(
            c2, h2, x2, self.upward2, self.lateral2)
        c, h = slstm.slstm(c1, c2, lstm_in1, lstm_in2)
        return c, h
