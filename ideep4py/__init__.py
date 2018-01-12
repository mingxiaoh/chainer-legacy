import numpy

from ideep4py._ideep4py import intVector  # NOQA

from ideep4py._ideep4py import mdarray  # NOQA
from ideep4py._ideep4py import mdarrayVector  # NOQA

from ideep4py._ideep4py import batchNormalization  # NOQA
from ideep4py._ideep4py import concat  # NOQA
from ideep4py._ideep4py import convolution2D  # NOQA
from ideep4py._ideep4py import convolution2DParam  # NOQA
from ideep4py._ideep4py import dropout  # NOQA
from ideep4py._ideep4py import linear  # NOQA
from ideep4py._ideep4py import localResponseNormalization  # NOQA
from ideep4py._ideep4py import localResponseNormalizationParam  # NOQA
from ideep4py._ideep4py import pooling2D  # NOQA
from ideep4py._ideep4py import pooling2DParam  # NOQA
from ideep4py._ideep4py import relu  # NOQA
from ideep4py._ideep4py import tanh  # NOQA

from ideep4py._ideep4py import basic_acc_sum  # NOQA
from ideep4py._ideep4py import basic_copyto  # NOQA

from ideep4py import cosim  # NOQA


# ------------------------------------------------------------------------------
# ideep4py.mdarray allocation
# ------------------------------------------------------------------------------
dat_array = 'd'  # data array
wgt_array = 'w'  # weight array


def array(x, itype=dat_array):
    """Create a :class:`ideep4py.mdarray` object according to ``x``.

    Args:
        array (numpy.ndarray or ideep4py.mdarray):
            if ``x`` is numpy.ndarray not in C contiguous, it will be
            converted to C contiguous before ideep4py.mdarray created.
        itype (=data_type): ideep4py.mdarray created is optimized according
            ``itype`` flag.

    Returns:
        Instance of :class:`ideep4py.mdarray`.

    """
    if isinstance(x, numpy.ndarray) and \
            x.dtype == numpy.dtype('float32'):
        if x.flags.contiguous is False:
            x = numpy.ascontiguousarray(x)
        return mdarray(x, itype)
    else:
        return x
