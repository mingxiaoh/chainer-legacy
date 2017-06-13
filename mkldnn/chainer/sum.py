import numpy
import chainer
from chainer.utils import type_check
from mkldnn.chainer.runtime import Engine
from mkldnn.compute_complex import *
# Most important thing
from mkldnn.api.support import *
import mkldnn.api.memory as m
import mkldnn.api.sum as sum
from mkldnn.mdarray import *

def mkl_sum_enabled(in_data):
    if chainer.should_use_mkldnn('>=auto') \
       and all(isinstance(xi, numpy.ndarray) or isinstance(xi, mkldnn.mdarray) for xi in in_data):
        return True
    else:
        return False

def _x_format(ndim):
    if ndim == 1:
        return m.memory.x
    if ndim == 2:
        return m.memory.nc
    elif ndim == 4:
        return m.memory.nchw
    else:
        return NotImplemented

def mkl_sum(xs):
    e = Engine()

    xarrays = ()
    xs_mpdl = m.mpd_list()
    xs_pl = ()
    scales = m.vectord()
    pl = primitive_list()
    for x in xs:
        xarray = array(x, _x_format(x.ndim), e)
        #xarray = reorder_if_must(xarray, xmpd, pl)
        xarrays += (xarray,)
        scales.push_back(1.0)
        xs_mpdl.push_back(xarray.memory.get_primitive_desc())
        xs_pl += (at(xarray.memory), )

    cc_pd = sum.primitive_desc(scales, xs_mpdl)
    y = mdarray(cc_pd.dst_primitive_desc())

    pl.push_back(sum.sum(cc_pd, xs_pl, y.memory))
    s = Stream()
    s.submit(pl)
    s.wait()

    return y
