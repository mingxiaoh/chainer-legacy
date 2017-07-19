import numpy

import chainer
from chainer import cuda
from chainer import mkld
from chainer import function
from chainer import utils
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()
    _mode = libcudnn.CUDNN_ACTIVATION_RELU

if mkld.available:
    ReLUMKLDNN = mkld.relu.ReLUMKLDNN


class ReLU(function.Function):

    """Rectified Linear Unit."""
    # TODO(beam2d): Implement in-place version.

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward_cpu(self, x):
        self.retain_inputs(())
        self.retain_outputs((0,))
        return utils.force_array(numpy.maximum(x[0], 0, dtype=x[0].dtype)),

    def forward_gpu(self, x):
        if (chainer.should_use_cudnn('==always') and
                x[0].flags.c_contiguous and
                (_cudnn_version >= 3000 or x[0].dtype != numpy.float16)):
            self._use_cudnn = True
            y = cudnn.activation_forward(x[0], _mode)
        else:
            self.retain_inputs(())
            self._use_cudnn = False
            y = cuda.cupy.maximum(x[0], 0)
        self.retain_outputs((0,))
        return y,

    def backward_cpu(self, x, gy):
        y = self.output_data[0]
        return utils.force_array(gy[0] * (y > 0)),

    def backward_gpu(self, x, gy):
        y = self.output_data[0]
        if chainer.should_use_cudnn('==always') and self._use_cudnn:
            gx = cudnn.activation_backward(x[0], y, gy[0], _mode)
        else:
            gx = cuda.elementwise(
                'T y, T gy', 'T gx',
                'gx = y > 0 ? gy : (T)0',
                'relu_bwd')(y, gy[0])
        return gx,


def relu(x):
    """Rectified Linear Unit function.

     .. math::`f(x)=\\max(0, x)`.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_n)`-shaped float array.

    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_n)`-shaped float array.

    .. admonition:: Example

        >>> x = np.array([[-1, 0], [2, -3], [-2, 1]], 'f')
        >>> np.any(x < 0)
        True
        >>> y = F.relu(x)
        >>> np.any(y.data < 0)
        False
        >>> y.shape
        (3, 2)

    """
    if not isinstance(x.data, cuda.ndarray) and \
       mkld.check_with_mkld((x, ), (2, 4)):
        func = ReLUMKLDNN()
        ret = func(x)
        if chainer.is_cosim():
            func.cosim_func = ReLU()
            numpy_result = func.cosim_func(x)
            func.cpu_cosim_verify_result(ret, numpy_result)
        return ret
    else:
        if chainer.mkld.available and \
           chainer.should_use_mkldnn('>=auto'):
            print('WARNING, relu inputs is not mdarray ', x.rank, type(x.data))
        return ReLU()(x)
