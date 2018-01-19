import chainer
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


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

        use_ideep = (chainer.ia.check_ideep_enabled()) and \
            (xs[0].ndim == 2 or xs[0].ndim == 4)
        if use_ideep:
            for x in xs:
                if not isinstance(x, chainer.ia.mdarray):
                    use_ideep = False
                    break

        if use_ideep:
            y = chainer.ia.acc_add(xs)
        else:
            y = xs[0] + xs[1]
            for x in xs[2:]:
                y += x
        return utils.force_array(y),

    def backward(self, indexes, gy):
        gys = ()
        for i in range(self.len):
            gys += gy[0],
        return gys


def accumulateAdd(xs):
    return AccumulateAdd().apply(xs)[0]
