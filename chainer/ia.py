import numpy
import contextlib

import chainer
from chainer import variable


available = False

try:
    import ideep4py
    from ideep4py import mdarray  # NOQA
    from ideep4py import intVector  # NOQA
    from ideep4py import mdarrayVector  # NOQA
    from ideep4py import batchNormalization  # NOQA
    from ideep4py import relu  # NOQA
    from ideep4py import convParam, convolution2D  # NOQA
    from ideep4py import poolingParam, pooling2D  # NOQA
    from ideep4py import concat  # NOQA
    from ideep4py import linearParam, linear  # NOQA
    from ideep4py import lrnParam, localResponseNormalization  # NOQA
    from ideep4py import dropout  # NOQA
    from ideep4py import cosim  # NOQA
    available = True
except Exception as ex:
    print('*** CPU acceleration is disabled: %s' % ex)

    class mdarray(object):
        pass


def is_enabled():
    # Check whether ideep4py installed

    return available


@contextlib.contextmanager
def disable():
    global available
    old = available
    available = False
    try:
        yield
    finally:
        available = old


def all_ready(inputs, check_with_ndim=(2, 4)):
    if not is_enabled():
        return False
    _inputs = [x.data if isinstance(x, variable.Variable)
               else x for x in inputs]

    # Check with ideep4py supported dimension of input data
    valid_ndim = False
    for ndim in check_with_ndim:
        valid_ndim = valid_ndim or _inputs[0].ndim == ndim

    if check_with_ndim and not valid_ndim:
        return False

    if isinstance(_inputs[0], mdarray):
        return True
    # Check whether ideep4py configured and used correctly
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

    return True


# ----------------------------------------------------------------------
# ideep4py mdarray allocation
# ---------------------------------------------------------------------
data = 'd'  # data array
weight = 'w'  # weight array


def array(x, itype=data):
    if not is_enabled():
        raise Exception("ideep4py is not installed coorectly")
    if isinstance(x, numpy.ndarray) and \
            x.dtype == numpy.dtype('float32'):
        if x.flags.contiguous is False:
            x = numpy.ascontiguousarray(x)
        return mdarray(x, itype)
    else:
        return x


def copyto(dst, src, casting='same_kind', where=None):
    if dst.shape != src.shape or dst.dtype != src.dtype:
        raise Exception("Can't copy, shape or type mismatch")
    if isinstance(src, numpy.ndarray):
        if src.flags.contiguous is False:
            src = numpy.ascontiguousarray(src)
    ideep4py.basic_copyto(dst, src)


def acc_add(xs):
    if xs[0].ndim == 2 or xs[0].ndim == 4:
        fast = True
    else:
        fast = False
    for x in xs:
        if not isinstance(x, mdarray):
            fast = False
            break
    if fast is True:
        return ideep4py.basic_acc_sum(xs)
    else:
        # y = sum(xs)
        y = xs[0] + xs[1]
        for x in xs[2:]:
            y += x
        if type(y) != type(xs[0]):
            y = numpy.asarray(y).astype(xs[0].dtype)
        return y
