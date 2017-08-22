import numpy

import chainer
from chainer import variable


available = False

try:
    import mkldnn
    from mkldnn.mdarray import mdarray
    from mkldnn.chainer import basic_math
    from mkldnn.chainer import fanout
    from mkldnn.chainer import runtime
    from mkldnn.chainer import sum
    # Modules listed depend on chainer.
    from mkldnn.chainer import avg_pooling_2d
    from mkldnn.chainer import bn
    from mkldnn.chainer import concat
    from mkldnn.chainer import convolution_2d
    from mkldnn.chainer import deconvolution_2d
    from mkldnn.chainer import linear
    from mkldnn.chainer import lrn
    from mkldnn.chainer import max_pooling_2d
    from mkldnn.chainer import pooling_2d
    from mkldnn.chainer import relu
    from mkldnn.chainer.optimization import training_forward_optimization

    available = True
except Exception as ex:
    print('WARNING: import mkldpy fails')
    error_info = ex

    class mdarray(object):
        pass


def all_ready(inputs, check_with_ndim):
    # Check whether mkldnn installed
    if not available:
        return False
    _inputs = [x.data if isinstance(x, variable.Variable)
               else x for x in inputs]

    if isinstance(_inputs[0], mdarray):
        return True
    # Check whether mkldnn configured and used correctly
    elif isinstance(_inputs[0], numpy.ndarray):
        _should_use_mkldnn = True

        for x in _inputs:
            _should_use_mkldnn = _should_use_mkldnn and \
                                 x.dtype == numpy.dtype('float32')
        if _should_use_mkldnn:
            _should_use_mkldnn = _should_use_mkldnn and \
                                 chainer.should_use_mkldnn('>=auto')
        if not _should_use_mkldnn:
            return False
    # cuda.ndarray
    else:
        return False

    # Check with mkldnn supported dimension of input data
    valid_ndim = False
    for ndim in check_with_ndim:
        valid_ndim = valid_ndim or _inputs[0].ndim == ndim

    if check_with_ndim and not valid_ndim:
        return False

    return True


def to_plain_array(params):
    assert(isinstance(params, tuple) or isinstance(params, list))

    _params = ()

    for p in params:
        if isinstance(p, mdarray):
            _params += (numpy.array(p), )
        else:
            _params += (p, )

    return _params
