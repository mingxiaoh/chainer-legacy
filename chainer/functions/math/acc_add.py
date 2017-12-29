import numpy
import chainer
from chainer.utils import type_check
from chainer import function_node


def accumulate_grad(gx, g_input):
    sum_gx = ()
    if isinstance(gx, tuple):
        sum_gx = gx
    elif gx is not None:
        sum_gx = gx,
    if isinstance(g_input, tuple):
        sum_gx += g_input
    elif g_input is not None:
        sum_gx += g_input,
    if len(sum_gx) == 0:
        sum_gx = None,
    return sum_gx


class AccumulateAdd(function_node.FunctionNode):

    def check_type_forward(self, in_types):
        for in_type in in_types:
            type_check.expect(
                in_types[0].dtype == in_type.dtype,
                in_types[0].shape == in_type.shape
            )

    def forward(self, xs):
        self.len = len(xs)
        if chainer.ia.check_ideep_enabled():
            y = chainer.ia.acc_add(xs)
        else:
            y = xs[0] + xs[1]
            for x in xs[2:]:
                y += x
            if type(y) != type(xs[0]):
                y = numpy.asarray(y).astype(xs[0].dtype)

        return y,

    def backward(self, indexes, gy):
        gys = ()
        for i in range(self.len):
            gys += gy[0],
        return gys


def accumulateAdd(xs):
    return AccumulateAdd().apply(xs)[0]
