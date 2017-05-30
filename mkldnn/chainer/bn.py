import collections
from chainer import function
from chainer import configuration
from chainer.utils import type_check
from chainer.utils import conv

from mkldnn.chainer.runtime import Engine
from mkldnn.compute_complex import *

# Most important thing
from mkldnn.api.support import *
import mkldnn.api.memory as m
import mkldnn.api.bn_forward as bn_forward
import mkldnn.api.bn_backward as bn_backward
from mkldnn.mdarray import *

class BnForward(ComputeComplex):
    cc_type = 'f'

    def __init__(self, inputs, eps=2e-5, mean=None, var=None,
            pos=None, e=Engine()):
        super(BnForward, self).__init__()

        x = inputs[0]
        if self.new:
            self._create_cc(inputs, eps, mean, var, e)
        else:
            self._reuse(inputs, mean, var)

    def _create_cc(self, inputs, eps, mean, var, e):
        self.eps = eps
        self.mean = None
        self.var = None
        self.w = None
        self.train = configuration.config.train
        x, gamma, beta = inputs[:3]

        self.x = array(x, m.memory.nchw, e)
        w = numpy.concatenate((gamma, beta), axis=0).reshape((2, -1))
        self.numpy_w = w
        self.w = array(w, m.memory.nc, e)
        scale_shift = True
        self.flags = use_scale_shift
        if mean is None:
            fwd_prop_kind = forward_training
            global_stats = False
        else:
            fwd_prop_kind = forward_scoring
            self.flags |= use_global_stats
            global_stats = True
            self.mean = array(mean, m.memory.x, e)
            self.var = array(var, m.memory.x, e)

        x_md = self.x.memory.get_primitive_desc().desc()
        cc_d = bn_forward.desc(fwd_prop_kind, x_md, eps, self.flags)
        cc_pd = bn_forward.primitive_desc(cc_d, e)
        y = mdarray(cc_pd.dst_primitive_desc())

        #TODO reorder weight
        #if scale_shift is True:
        #    w = mdarray(cc_pd.weights_primitive_desc())
        if scale_shift is True and global_stats is False:
            self.mean = mdarray(cc_pd.mean_primitive_desc())
            self.var = mdarray(cc_pd.variance_primitive_desc())

        if (not configuration.config.train) and (not global_stats):
            if scale_shift is True:
                print('1----------------------------')
                bnf = bn_forward.batch_normalization_forward(cc_pd, at(self.x.memory), at(self.w.memory), y.memory)
            else:
                print('2----------------------------')
                bnf = bn_forward.batch_normalization_forward(cc_pd, at(self.x.memory), y.memory)
        elif global_stats is True:
            if scale_shift is True:
                print('3----------------------------')
                bnf = bn_forward.batch_normalization_forward(cc_pd, at(self.x.memory), at(self.mean.memory),
                        at(self.var.memory), at(self.w.memory), y.memory)
            else:
                print('4----------------------------')
                bnf = bn_forward.batch_normalization_forward(cc_pd, at(self.x.memory), self.mean.memory,
                        self.var.memory, y.memory)
        else:
            if scale_shift is True:
                print('5----------------------------')
                bnf = bn_forward.batch_normalization_forward(cc_pd, at(self.x.memory), at(self.w.memory),
                        y.memory, self.mean.memory, self.var.memory)
            else:
                print('6----------------------------')
                bnf = bn_forward.batch_normalization_forward(cc_pd, at(self.x.memory),
                        y.memory, self.mean.memory, self.var.memory)

        self.dag_.push_back(bnf)
        self._hint = cc_pd
        self.outputs = y, self.flags, self.mean, self.var

    def _reuse(self, inputs, mean=None, var=None):
        x, gamma, beta = inputs[:3]
        reuse_buffer(self.x, x)
        if mean is not None:
            reuse_buffer(self.mean, mean)
        if var is not None:
            reuse_buffer(self.var, var)
        w = numpy.concatenate((gamma, beta), axis=0).reshape((2, -1))
        reuse_buffer(self.w, w)

    def match(self, inputs, eps, mean=None, var=None):
        x = inputs[0]
        if (self.x.shape != x.shape) or (self.eps != eps):
            print('WARNING:bn forward, shape or eps mismatch ', self.x.shape, x.shape, self.eps, eps)
            return False
        if self.train != configuration.config.train:
            print('WARNING:bn forward, config.train mismatch ', self.train, configuration.config.train)
            return False
        if (mean is not None) and ((self.flags & use_global_stats) == 0):
            print('WARNING:bn forward, mean or flags mismatch ', mean, self.flags)
            return False
        return True

class BnBackward(ComputeComplex):
    cc_type = 'bd'

    def __init__(self, inputs, gy, hint, flags, eps, mean, var,
            pos=None, e=Engine()):
        super(BnBackward, self).__init__()

        x = inputs[0]
        if self.new:
            self._create_cc(inputs, gy, hint, flags, eps, mean, var, e)
        else:
            self._reuse(inputs, gy, mean, var)

    def _create_cc(self, inputs, gy, hint, flags, eps, mean, var, e):
        self.train = configuration.config.train
        self.flags = flags
        self.eps = eps
        x, gamma, beta = inputs[:3]
        self.x = array(x, m.memory.nchw, e)
        x_md = self.x.memory.get_primitive_desc().desc()
        gy = array(gy, m.memory.nchw, e)
        gy_md = gy.memory.get_primitive_desc().desc()
        cc_d = bn_backward.desc(backward, gy_md, x_md, eps, flags)
        cc_pd = bn_backward.primitive_desc(cc_d, e, hint)

        gx = mdarray(self.x.memory.get_primitive_desc())
        if flags & use_scale_shift:
            w = numpy.concatenate((gamma, beta), axis=0).reshape((2, -1))
            self.w = array(w, m.memory.nc, e)
            self.mean = array(mean, m.memory.x, e)
            self.var = array(var, m.memory.x, e)
            self.gw = mdarray(cc_pd.diff_weights_primitive_desc())
            bwd_p = bn_backward.batch_normalization_backward(cc_pd, at(self.x.memory), at(self.mean.memory),
                    at(self.var.memory), at(gy.memory), at(self.w.memory), gx.memory, self.gw.memory)
        else:
            bwd_p = bn_backward.batch_normalization_backward(cc_pd, at(self.x.memory), at(self.mean.memory),
                    at(self.var.memory), at(gy.memory), gx.memory)

        self.dag_.push_back(bwd_p)
        self._hint = hint
        self.gy = gy
        self.outputs = gx, self.gw

    def _reuse(self, inputs, gy, mean=None, var=None):
        x, gamma, beta = inputs[:3]
        reuse_buffer(self.x, x)
        reuse_buffer(self.gy, gy)
        if mean is not None:
            reuse_buffer(self.mean, mean)
        if var is not None:
            reuse_buffer(self.var, var)
        w = numpy.concatenate((gamma, beta), axis=0).reshape((2, -1))
        reuse_buffer(self.w, w)

    def match(self, inputs, gy, hint, flags, eps, mean, var,
            pos=None, e=Engine()):
        if self.train != configuration.config.train:
            print('WARNING:bn backward, config.train mismatch ', self.train, configuration.config.train)
            return False
        x = inputs[0]
        if (self.x.shape != x.shape) or (self.eps != eps):
            return False
        return True
