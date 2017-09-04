import os
import numpy

from mkldnn.mdarray import mdarray

from chainer import variable
from chainer import configuration

from chainer.configuration import config
from chainer.configuration import global_config

# export CHAINER_ENABLE_COSIM=0
global_config.cosim = bool(int(os.environ.get('CHAINER_ENABLE_COSIM', '0')))
# export CHAINER_ENABLE_COSIM_CONTINUE=0
global_config.cosim_continue = bool(int(os.environ.get('CHAINER_ENABLE_COSIM_CONTINUE', '0')))

if global_config.cosim is True:
    variable.RANK_START = 1


def is_cosim_continue():
    """Get the cosim continue mode

    Returns:
        bool: Return ``True`` if continue after a failure.
    """
    return config.cosim_continue


def is_cosim():
    """Get the cosim mode.

    Returns:
        bool: Return ``True`` if Chainer is in cosim mode.
    """
    return config.cosim


def plain_array(params):
    assert isinstance(params, tuple) \
            or isinstance(params, list) \
            or isinstance(params, mdarray) \
            or isinstance(params, numpy.ndarray) \
            or isinstance(params, variable.Variable)

    _params = ()

    if isinstance(params, variable.Variable):
        return (numpy.array(params.data), )
    elif isinstance(params, numpy.ndarray):
        return (params, )
    elif isinstance(params, mdarray):
        return (numpy.array(params), )

    for p in params:
        if isinstance(p, variable.Variable):
            p = numpy.array(p.data)
        if isinstance(p, mdarray):
            _params += (numpy.array(p), )
        else:
            _params += (p, )

    return _params

