import numpy

import chainer
from chainer import variable


available = False

try:
    import dnn._dnn
    from dnn._dnn import mdarray

    available = True
except Exception as ex:
    print('*** CPU acceleration is disabled: %s' % ex)

    class mdarray(object):
        pass


def is_enabled():
    # Check whether ideep installed

    return available


def all_ready(inputs, check_with_ndim):
    if not is_enabled():
        return False
    _inputs = [x.data if isinstance(x, variable.Variable)
               else x for x in inputs]

    if isinstance(_inputs[0], mdarray):
        return True
    # Check whether ideep configured and used correctly
    elif isinstance(_inputs[0], numpy.ndarray):
        _should_use_ideep = True

        for x in _inputs:
            _should_use_ideep = _should_use_ideep and \
                                 x.dtype == numpy.dtype('float32')
        if _should_use_ideep:
            _should_use_ideep = _should_use_ideep and \
                                 chainer.should_use_ideep('>=auto')
        if not _should_use_ideep:
            return False
    # cuda.ndarray
    else:
        return False

    # Check with ideep supported dimension of input data
    valid_ndim = False
    for ndim in check_with_ndim:
        valid_ndim = valid_ndim or _inputs[0].ndim == ndim

    if check_with_ndim and not valid_ndim:
        return False
    return True
