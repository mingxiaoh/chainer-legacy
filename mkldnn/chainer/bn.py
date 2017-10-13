from chainer import function
from chainer import configuration
from chainer.utils import type_check

from mkldnn.chainer import cosim, is_cosim
from mkldnn.chainer.runtime import Engine
from mkldnn.compute_complex import reorder_if_must
from mkldnn.compute_complex import reuse_buffer
from mkldnn.compute_complex import ComputeComplex
from mkldnn.array import array
import numpy

# Most important thing
from mkldnn.api.support import use_scale_shift
from mkldnn.api.support import forward_training
from mkldnn.api.support import forward_scoring
from mkldnn.api.support import use_global_stats
from mkldnn.api.support import at
from mkldnn.api.support import backward
import mkldnn.api.memory as m
import mkldnn.api.bn_forward as bn_forward
import mkldnn.api.bn_backward as bn_backward
from mkldnn.mdarray import mdarray
import mkldnn.api.cosim_dump as cdump
from mkldnn.api.cosim_dump import *


def _xhat(x, mean, std, expander):
    x_mu = x - mean[expander]
    x_mu /= std[expander]
    return x_mu


class BnForward(ComputeComplex):
    cc_type = 'f'

    def __init__(self, inputs, eps=2e-5, mean=None, var=None,
                 pos=None, e=Engine()):
        super(BnForward, self).__init__()

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

        fmt_desired = m.get_desired_format(x.shape[1])
        x = array(x, m.memory.nchw, e)
        # x = array(x, fmt_desired, e)

        assert x.dtype == numpy.dtype('float32')
        x_desired_md = m.desc(x.shape, m.memory.f32, fmt_desired)
        x_desired_mpd = m.primitive_desc(x_desired_md, e)
        outputs = reorder_if_must(x, x_desired_mpd, e, self.dag_)
        if len(outputs) == 2:
            self.x, self.itm_arr = outputs[:2]
            self.x_src = x
        else:
            self.x = outputs[0]
            self.x_src = x

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

        # TODO reorder weight
        # if scale_shift is True:
        #    w = mdarray(cc_pd.weights_primitive_desc())
        if scale_shift is True and global_stats is False:
            self.mean = mdarray(cc_pd.mean_primitive_desc())
            self.var = mdarray(cc_pd.variance_primitive_desc())

        if (not configuration.config.train) and (not global_stats):
            if scale_shift is True:
                bnf = bn_forward.batch_normalization_forward(cc_pd, at(self.x.memory), at(self.w.memory), y.memory)
            else:
                bnf = bn_forward.batch_normalization_forward(cc_pd, at(self.x.memory), y.memory)
        elif global_stats is True:
            if scale_shift is True:
                bnf = bn_forward.batch_normalization_forward(cc_pd, at(self.x.memory), at(self.mean.memory),
                                                             at(self.var.memory), at(self.w.memory), y.memory)
            else:
                bnf = bn_forward.batch_normalization_forward(cc_pd, at(self.x.memory), self.mean.memory,
                                                             self.var.memory, y.memory)
        else:
            if scale_shift is True:
                bnf = bn_forward.batch_normalization_forward(cc_pd, at(self.x.memory), at(self.w.memory),
                                                             y.memory, self.mean.memory, self.var.memory)
            else:
                bnf = bn_forward.batch_normalization_forward(cc_pd, at(self.x.memory),
                                                             y.memory, self.mean.memory, self.var.memory)

        self.dag_.push_back(bnf)
        self._hint = cc_pd
        self.outputs = y, self.flags, self.mean, self.var

    def _reuse(self, inputs, mean=None, var=None):
        x, gamma, beta = inputs[:3]
        reuse_buffer(self.x_src, x)
        if mean is not None:
            reuse_buffer(self.mean, mean)
        if var is not None:
            reuse_buffer(self.var, var)
        w = numpy.concatenate((gamma, beta), axis=0).reshape((2, -1))
        reuse_buffer(self.w, w)

    def match(self, inputs, eps, mean=None, var=None):
        x = inputs[0]
        if (self.x.shape != x.shape) or (self.eps != eps):
            # print('WARNING:bn forward, shape or eps mismatch ', self.x.shape, x.shape, self.eps, eps)
            return False
        if (isinstance(x, mdarray) and (x is not self.x_src)):
            return False

        if self.train != configuration.config.train:
            # print('WARNING:bn forward, config.train mismatch ', self.train, configuration.config.train)
            return False
        if (mean is not None) and ((self.flags & use_global_stats) == 0):
            # print('WARNING:bn forward, mean or flags mismatch ', mean, self.flags)
            return False
        return True


class BnBackward(ComputeComplex):
    cc_type = 'bd'

    def __init__(self, inputs, fwd_x, gy, hint, flags, eps, mean, var,
                 pos=None, e=Engine()):
        super(BnBackward, self).__init__()

        if self.new:
            self._create_cc(inputs, fwd_x, gy, hint, flags, eps, mean, var, e)
        else:
            self._reuse(inputs, gy, mean, var)

    def _create_cc(self, inputs, fwd_x, gy, hint, flags, eps, mean, var, e):
        self.train = configuration.config.train
        self.flags = flags
        self.eps = eps
        x, gamma, beta = inputs[:3]
        # self.x = array(x, m.memory.nchw, e)
        self.x = fwd_x
        x_mpd = self.x.memory.get_primitive_desc()
        x_md = x_mpd.desc()
        gy = array(gy, m.memory.nchw, e)
        outputs = reorder_if_must(gy, x_mpd, e, self.dag_)
        if len(outputs) == 2:
            self.gy_src = gy
            gy, self.itm_arr = outputs[:2]
        else:
            self.gy_src = gy
            gy = outputs[0]

        gy_md = gy.memory.get_primitive_desc().desc()
        cc_d = bn_backward.desc(backward, gy_md, x_md, eps, flags)
        cc_pd = bn_backward.primitive_desc(cc_d, e, hint)

        # gx = mdarray(self.x.memory.get_primitive_desc(), gy.memory)
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
        reuse_buffer(self.gy_src, gy)
        if mean is not None:
            reuse_buffer(self.mean, mean)
        if var is not None:
            reuse_buffer(self.var, var)
        w = numpy.concatenate((gamma, beta), axis=0).reshape((2, -1))
        reuse_buffer(self.w, w)

    def match(self, inputs, fwd_x, gy, hint, *args):
        if self.train != configuration.config.train:
            print('WARNING:bn backward, config.train mismatch ', self.train, configuration.config.train)
            return False
        return (hint is self._hint)


class BnMKLDNN(function.Function):

    def __init__(self, eps=2e-5, mean=None, var=None, decay=0.9):
        self.running_mean = mean
        self.running_var = var

        self.eps = eps
        self.mean_cache = None
        self.decay = decay

        if is_cosim():
            from chainer.functions.normalization.batch_normalization import BatchNormalizationFunction
            self.mean_orig = mean
            self.cosim_func = BatchNormalizationFunction(eps, mean, var, decay)

    def check_type_forward(self, in_types):
        n_in = type_check.eval(in_types.size())
        if n_in != 3 and n_in != 5:
            raise type_check.InvalidType(
                '%s or %s' % (in_types.size() == 3, in_types.size() == 5),
                '%s == %s' % (in_types.size(), n_in))
        x_type, gamma_type, beta_type = in_types[:3]
        M = type_check.eval(gamma_type.ndim)
        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim >= gamma_type.ndim + 1,
            x_type.shape[1:1 + M] == gamma_type.shape,
            # TODO(beam2d): Check shape
            gamma_type.dtype == x_type.dtype,
            beta_type.dtype == x_type.dtype,
            gamma_type.shape == beta_type.shape,
        )
        if len(in_types) == 5:
            mean_type, var_type = in_types[3:]
            type_check.expect(
                mean_type.dtype == x_type.dtype,
                mean_type.shape == gamma_type.shape,
                var_type.dtype == x_type.dtype,
                var_type.shape == gamma_type.shape,
            )

    def forward(self, inputs):
        x, gamma, beta = inputs[:3]
        if configuration.config.train:
            if self.running_mean is None:
                self.running_mean = numpy.zeros_like(gamma)
                self.running_var = numpy.zeros_like(gamma)
            else:
                self.running_mean = numpy.array(self.running_mean)
                self.running_var = numpy.array(self.running_var)
        elif len(inputs) == 5:
            self.fixed_mean = inputs[3]
            self.fixed_var = inputs[4]

        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        gamma = gamma[expander]
        beta = beta[expander]

        if (isinstance(x, mdarray)
                or (isinstance(x, numpy.ndarray)
                    and x.dtype == numpy.dtype('float32')
                    and (x.ndim == 2 or x.ndim == 4))):
            outputs = self.forward_cpu(inputs)
            y = outputs[0]
            self.flags = outputs[1]
            if configuration.config.train:
                mean = outputs[2]
                var = outputs[3]
        else:
            if configuration.config.train:
                axis = (0,) + tuple(range(head_ndim, x.ndim))
                mean = x.mean(axis=axis)
                var = x.var(axis=axis)
                var += self.eps
            else:
                mean = self.fixed_mean
                var = self.fixed_var + self.eps

            self.std = numpy.sqrt(var, dtype=var.dtype)
            self.x_hat = _xhat(x, mean, self.std, expander)
            y = gamma * self.x_hat
            y += beta

        if configuration.config.train:
            # Update running statistics:
            m = x.size // gamma.size
            adjust = m / max(m - 1., 1.)  # unbiased estimation
            self.running_mean *= self.decay
            temp_ar = numpy.array(mean)
            temp_ar *= (1 - self.decay)
            self.running_mean += temp_ar
            del temp_ar
            self.running_var *= self.decay
            temp_ar = numpy.array(var)
            temp_ar *= (1 - self.decay) * adjust
            self.running_var += temp_ar
            del temp_ar

        return y,

    def backward(self, inputs, grad_outputs):
        x, gamma = inputs[:2]
        gy = grad_outputs[0]
        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        m = gamma.dtype.type(x.size // gamma.size)
        axis = (0,) + tuple(range(head_ndim, x.ndim))
        if len(inputs) == 5:
            # This case is unlikely to be used in practice and so does not
            # need to be optimized for performance.
            mean = inputs[3]
            var = inputs[4]
            std = numpy.sqrt(var, dtype=var.dtype)
            gs = gamma / std
            gbeta = gy.sum(axis=axis)
            x_hat = _xhat(x, mean, std, expander)
            ggamma = (gy * x_hat).sum(axis=axis)
            gmean = -gs * gbeta
            gvar = -0.5 * gamma / var * ggamma
            gx = gs[expander] * gy

            return gx, ggamma, gbeta, gmean, gvar

        # Note: If length of inputs is not 5, we must be in train mode.
        assert configuration.config.train
        if (isinstance(x, mdarray)
                or (isinstance(x, numpy.ndarray)
                    and x.dtype == numpy.dtype('float32')
                    and (x.ndim == 2 or x.ndim == 4))):
            outputs = self.backward_cpu(inputs, gy)
            gx, ggamma, gbeta = outputs[:3]
        else:
            gbeta = gy.sum(axis=axis)
            ggamma = (gy * self.x_hat).sum(axis=axis)
            gx = (gamma / self.std)[expander] * (
                gy - (self.x_hat * ggamma[expander] + gbeta[expander]) / m)

        return gx, ggamma, gbeta

    def forward_cpu(self, inputs):
        self.expand_dim = False
        x = inputs[0]
        if x.ndim == 2:
            self.expand_dim = True
            x = x[:, :, None, None]
            inputs = (x,) + inputs[1:]

        if configuration.config.train:
            cc = BnForward(
                inputs, self.eps, None, None,
                pos=(self.rank, self.fanout))
        else:
            cc = BnForward(
                inputs, self.eps, self.fixed_mean, self.fixed_var,
                pos=(self.rank, self.fanout))

        self.hint = cc.hint
        self.fwd_x = cc.x
        outputs = cc.execute_on()
        if configuration.config.train:
            self.mkl_mean = outputs[2]
            self.mkl_var = outputs[3]
        y = outputs[0]
        if self.expand_dim:
            assert y.ndim == 4
            y = numpy.squeeze(y, axis=(2, 3))
        outputs = (y,) + outputs[1:]

        cosim.cosim_verify(self, (y, ), inputs)
        return outputs

    def backward_cpu(self, inputs, gy):
        expand_dim = False
        x = inputs[0]
        if x.ndim == 2:
            expand_dim = True
            x = x[:, :, None, None]
            gy = gy[:, :, None, None]
        inputs = (x,) + inputs[1:]

        gy_orig = None
        if is_cosim():
            import copy
            gy_orig = copy.deepcopy(gy)

        if configuration.config.train:
            mean = self.mkl_mean
            var = self.mkl_var
        else:
            mean = self.fixed_mean
            var = self.fixed_var
        cc = BnBackward(
            inputs, self.fwd_x, gy, self.hint, self.flags,
            self.eps, mean, var,
            pos=(self.rank, self.fanout))

        outputs = cc.execute_on()
        gx = outputs[0]
        gx.reset_buf_order()
        ggamma = outputs[1][0]
        gbeta = outputs[1][1]
        if expand_dim:
            assert gx.ndim == 4
            gx = numpy.squeeze(gx, axis=(2, 3))

        cosim.cosim_verify(self, (gx, ggamma, gbeta), inputs, (gy_orig, ))
        return gx, ggamma, gbeta

    def dump_to_file(self, inputs, grads=None):
        cd = None
        if grads is None:
            cd = cdump.cosim_dump(cdump_op_bn_forward)
        else:
            cd = cdump.cosim_dump(cdump_op_bn_backward)

        e = Engine()

        x, gamma, beta = inputs[:3]
        x = array(inputs[0], m.memory.nchw, e)
        cd.dump_memory(cdump_src_memory, x.memory)

        w = numpy.concatenate((gamma, beta), axis=0).reshape((2, -1))
        w = array(w, m.memory.nc, e)
        cd.dump_memory(cdump_weight_memory, w.memory)

        if grads is not None:
            gy = array(grads[0], m.memory.nchw, e)
            cd.dump_memory(cdump_diff_dst_memory, gy.memory)

        # Always True
        scale_shift = 1
        global_stats = 1
        fwd_prop_kind = forward_scoring
        if self.mean_orig is None:
            fwd_prop_kind = forward_training
            global_stats = 0

        cd.dump_int_parms(cdump_bn_int_parms, 3, fwd_prop_kind, global_stats, scale_shift)
        cd.dump_double_parms(cdump_bn_doulbe_parms, 1, self.eps)

