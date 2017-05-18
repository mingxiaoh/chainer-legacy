import os
import sys

# For C++ extension to work
sys.setdlopenflags(sys.getdlopenflags() | os.RTLD_GLOBAL)

# API lift
from mkldnn import api
from mkldnn import chainer
from mkldnn import compute_complex

from mkldnn.mdarray import mdarray

from mkldnn.chainer.fanout import *
