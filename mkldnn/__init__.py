import sys

# For C++ extension to work
if sys.version_info.major >= 3 and sys.version_info.minor >= 3:
    import os
    ld_global_flag = os.RTLD_GLOBAL
else:
    import ctypes
    ld_global_flag = ctypes.RTLD_GLOBAL
sys.setdlopenflags(sys.getdlopenflags() | ld_global_flag)

# API lift
from mkldnn import api
from mkldnn import chainer
from mkldnn import compute_complex

from mkldnn.mdarray import mdarray

from mkldnn.chainer.fanout import *
