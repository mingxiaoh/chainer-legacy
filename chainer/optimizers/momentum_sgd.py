import numpy

from chainer import cuda
from chainer import optimizer
from chainer import mkld

if mkld.mkldnn_enabled:
    mkldnn = mkld.mkldnn


class MomentumSGD(optimizer.GradientMethod):

    """Classical momentum SGD."""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.scale = numpy.empty((3,), dtype=numpy.float64)  # scale=[1, -lr, momentum]
        self.scale[0] = 1
        self.scale[1] = -lr
        self.scale[2] = momentum

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device(param.data):
            state['v'] = xp.zeros_like(param.data)

    def update_one_cpu(self, param, state):
        """
        data = data + v*momentum - lr*grad
        """
        v = state['v']
        if mkld.enable_sgdF((param.data, param.grad, v)):
            input = ()
            for xi in (param.data, param.grad, v):
                if xi.flags.contiguous is False:
                    input += (xi.copy().astype(xi.dtype),)
                else:
                    input += (xi,)

            mkldnn_sum = mkldnn.Sum_F32()
            ndim = param.data.ndim
            if ndim == 1:
                mkldnn_sum.sum1d_diff(input, param.data, self.scale)
            elif ndim == 2:
                mkldnn_sum.sum2d_diff(input, param.data, self.scale)
            elif ndim == 4:
                mkldnn_sum.sum4d_diff(input, param.data, self.scale)
            else:
                print("Not Implemented dims ", ndim)
        else:
            v *= self.momentum
            v -= self.lr * param.grad
            param.data += v

    def update_one_gpu(self, param, state):
        cuda.elementwise(
            'T grad, T lr, T momentum',
            'T param, T v',
            '''v = momentum * v - lr * grad;
               param += v;''',
            'momentum_sgd')(param.grad, self.lr, self.momentum,
                            param.data, state['v'])
