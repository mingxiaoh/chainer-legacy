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
    from mkldnn.chainer import linear
    from mkldnn.chainer import lrn
    from mkldnn.chainer import max_pooling_2d
    from mkldnn.chainer import pooling_2d
    from mkldnn.chainer import relu

    available = True
except Exception as ex:
    error_info = ex

    class mdarray(object):
        pass


def check_with_mkld(inputs, check_with_ndim):
    # Check whether mkldnn installed
    if not available:
        return False

    _inputs = [x.data if isinstance(x, variable.Variable)
               else x for x in inputs]

    # Check whether mkldnn configured and used correctly
    if not isinstance(_inputs[0], mdarray):
        _should_use_mkldnn = True

        for x in _inputs:
            _should_use_mkldnn = _should_use_mkldnn and \
                                 x.dtype == numpy.dtype('float32')
        if _should_use_mkldnn:
            _should_use_mkldnn = _should_use_mkldnn and \
                                 chainer.should_use_mkldnn('>=auto')
        if not _should_use_mkldnn:
            return False

    # Check with mkldnn supported dimension of input data
    valid_ndim = False
    for ndim in check_with_ndim:
        valid_ndim = valid_ndim or _inputs[0].ndim == ndim

    if check_with_ndim and not valid_ndim:
        return False

    return True
