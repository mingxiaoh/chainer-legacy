import numpy

from chainer import function_node
import chainer.functions
from chainer.utils import type_check

from chainer import ideepy
import dnn._dnn
from dnn._dnn import mdarray, linear_param_t, Linear_Py_F32



class LinearFunction(function_node.FunctionNode):

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim == 2,
            w_type.ndim == 2,
            x_type.shape[1] == w_type.shape[1],
        )
        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )
    def forward_ia(self, inputs):
        x = inputs[0]
        W = inputs[1]
        b = inputs[2] if len(inputs) == 3 else None
        # create linear parameter
        # for IA specific
        lp = ideepy.linear_param_t()
        lp.with_bias = True if len(inputs) == 3 else False

        if isinstance(W, numpy.ndarray):
            lp.with_weights_opt = False
        elif isinstance(W, ideepy.mdarray):
            #if weight is empty, we can do weights opt(pass optimized weight back)
            lp.with_weights_opt = True
        (x, W) = ideepy.to_mdarray((x, W))

        if lp.with_bias:
            (b, ) = ideepy.to_mdarray((b, ))
            y = Linear_Py_F32.Forward(x, W, b, lp)
        else:
            y = Linear_Py_F32.Forward(x, W, None, lp)
        self.retain_inputs((0, 1)) # b is not retained 
        return y,


    def forward(self, inputs):
        x = inputs[0]
        W = inputs[1]
        if (ideepy.all_ready(inputs, (2, 4))):
            return self.forward_ia(inputs)
 
        if not type_check.same_types(*inputs):
            raise ValueError('numpy and cupy must not be used together\n'
                             'type(W): {0}, type(x): {1}'
                             .format(type(W), type(x)))

        # NumPy raises an error when the array is not contiguous.
        # See: https://github.com/chainer/chainer/issues/2744
        # TODO(niboshi): Remove this code when NumPy is fixed.
        if (isinstance(x, numpy.ndarray) and
                not (x.flags.c_contiguous or x.flags.f_contiguous) and
                1 in x.shape):
            x = numpy.ascontiguousarray(x)

        y = x.dot(W.T).astype(x.dtype, copy=False)
        if len(inputs) == 3:
            b = inputs[2]
            y += b
        self.retain_inputs((0, 1))  # b is not retained
        return y,

    def backward(self, indexes, grad_outputs):
        x, W = self.get_retained_inputs()
        gy, = grad_outputs
        ret = []
        if 0 in indexes:
            gx, = LinearGradData(self).apply((gy, W))
            #gx = linear(gy, W.T)
            ret.append(chainer.functions.cast(gx, x.dtype))
        if 1 in indexes:
            gW, = LinearGradWeight(self).apply((x, gy))
            #gW = linear(gy.T, x.T)
            ret.append(chainer.functions.cast(gW, W.dtype))
        if 2 in indexes:
            gb = chainer.functions.sum(gy, axis=0)
            ret.append(gb)
        return ret


class LinearGradData(function_node.FunctionNode):
    def __init__(self, linear):
        return 
    def forward_ia(self, inputs):
        self.retain_inputs((0, 1))
        gy, W = inputs
        #create linear parameter
        #for IA specific
        lp = linear_param_t()
        lp.with_bias = False
        lp.src_ndims = 2
        
        (gy, W) = ideepy.to_mdarray((gy, W))
        gx = Linear_Py_F32.BackwardData(W, gy, lp)
        return gx,

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        gy, W = inputs
        if (ideepy.all_ready(inputs, (2, 4))):
            return self.forward_ia(inputs)
        gx = linear(gy, W.T)
        return gx.data,

    def backward(self, indexes, grad_outputs):
        gy, W = self.get_retained_inputs()
        ggx, = grad_outputs
        
        ret = []
        if 0 in indexes:
            ggy = linear(ggx, W)
            ret.append(chainer.functions.cast(ggy, gy.dtype))
        if 1 in indexes:
            gw, = LinearGradWeight(self).apply((ggx, gy))#linear(gy.T, ggx.T)
            ret.append(chainer.functions.cast(gw, W.dtype))
        if 2 in indexes:
            ggb = chainer.functions.sum(ggx, axis=0)
            ret.append(ggb)
        return ret

class LinearGradWeight(function_node.FunctionNode):
    
    def __init__(self, linear):
        W_node = linear.inputs[1]
        self.W_dtype = W_node.dtype

    def forward_ia(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs
        #create linear parameter
        #for IA specific
        lp = linear_param_t()
        lp.with_bias = False
        lp.src_ndims = 2
        (x, gy) = ideepy.to_mdarray((x, gy))
        #only calculate gW, no gb
        (gW,) = Linear_Py_F32.BackwardWeights(x, gy, lp)
        return gW,

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        if (ideepy.all_ready(inputs, (2, 4)) and self.W_dtype == numpy.dtype('float32')):
            return self.forward_ia(inputs)
        x, gy = inputs
        gW = linear(gy.T, x.T)
        return gW.data,
    
    def backward(self, indexes, grad_outputs):
        x, gy = self.get_retained_inputs()
        ggW, = grad_outputs

        ret = []
        if 0 in indexes:
            gx, = LinearGradData(self).apply((gy, ggW))
            #gx = linear(gy, ggw.T)
            ret.append(chainer.functions.cast(gx, x.dtype))
        if 1 in indexes:
            ggy = linear(x, ggW)
            ret.append(chainer.functions.cast(ggy, gy.dtype))
        return ret
            

def linear(x, W, b=None):
    """Linear function, or affine transformation.

    It accepts two or three arguments: an input minibatch ``x``, a weight
    matrix ``W``, and optionally a bias vector ``b``. It computes

    .. math:: Y = xW^\\top + b.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable, which is a :math:`(s_B, s_1, \
            s_2, ..., s_n)`-shaped float array. Its first dimension
            :math:`(s_B)` is assumed to be the *minibatch dimension*. The
            other dimensions are treated as concatenated one dimension whose
            size must be :math:`(s_1 * ... * s_n = N)`.
        W (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Weight variable of shape :math:`(M, N)`,
            where :math:`(N = s_1 * ... * s_n)`.
        b (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Bias variable (optional) of shape
            :math:`(M,)`.

    Returns:
        ~chainer.Variable: Output variable. A float array with shape
        of :math:`(s_B, M)`.

    .. seealso:: :class:`~chainer.links.Linear`

    .. admonition:: Example

        >>> x = np.random.uniform(0, 1, (3, 4)).astype('f')
        >>> W = np.random.uniform(0, 1, (5, 4)).astype('f')
        >>> b = np.random.uniform(0, 1, (5,)).astype('f')
        >>> y = F.linear(x, W, b)
        >>> y.shape
        (3, 5)

    """
    if x.ndim > 2:
        x = x.reshape(len(x), -1)
    
    if b is None:
        args = x, W
    else:
        args = x, W, b

    y, = LinearFunction().apply(args)
    return y
