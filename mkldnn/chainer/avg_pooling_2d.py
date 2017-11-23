import mkldnn.api.memory as m
import mkldnn.api.cosim_dump as cdump

from mkldnn.chainer import cosim, is_cosim
from mkldnn.chainer.pooling_2d import Pooling2DMKLDNN
from mkldnn.chainer.pooling_2d import Pooling2DForward
from mkldnn.chainer.pooling_2d import Pooling2DBackward
from mkldnn.api.support import pooling_avg_include_padding
from mkldnn.chainer.runtime import Engine
from mkldnn.array import array
from mkldnn.api.cosim_dump import *


class AvgPooling2DMKLDNN(Pooling2DMKLDNN):

    """Average pooling over a set of 2d planes."""

    def __init__(self, ksize, stride=None, pad=0, cover_all=True):
        super(AvgPooling2DMKLDNN, self).__init__(ksize, stride, pad, cover_all)

        if is_cosim():
            from chainer.functions.pooling.average_pooling_2d import AveragePooling2D
            self.cosim_func = AveragePooling2D(ksize, stride, pad, cover_all)

    def forward_cpu(self, x):
        cc = Pooling2DForward(x, pooling_avg_include_padding, ksize=(self.kh, self.kw),
                              stride=(self.sy, self.sx),
                              pad=(self.ph, self.pw), cover_all=self.cover_all,
                              pos=(self.rank, self.fanout))

        self.hint = cc.hint
        self.ws = cc.ws
        y, = cc.execute_on()
        self.y = y

        cosim.cosim_verify(self, (y, ), x)
        return y,

    def backward_cpu(self, x, gy):
        cc = Pooling2DBackward(x, gy[0], self.hint, self.y, self.ws, pooling_avg_include_padding,
                               ksize=(self.kh, self.kw),
                               stride=(self.sy, self.sx),
                               pad=(self.ph, self.pw), cover_all=self.cover_all,
                               pos=(self.rank, self.fanout))
        gx, = cc.execute_on()

        cosim.cosim_verify(self, (gx, ), x, gy)
        return gx,

    def dump_to_file(self, inputs, grads=None):
        cd = None
        if grads is None:
            cd = cdump.cosim_dump(cdump_op_avg_pooling_forward)
        else:
            cd = cdump.cosim_dump(cdump_op_avg_pooling_backward)

        x = array(inputs[0], m.memory.nchw, Engine())
        cd.dump_memory(cdump_src_memory, x.memory)

        if grads is not None:
            gy = array(grads[0], m.memory.nchw, Engine())
            cd.dump_memory(cdump_diff_dst_memory, gy.memory)

        cd.dump_int_parms(cdump_avg_pooling_int_parms, 8,
                          self.kh, self.kw, self.sy, self.sx, self.ph, self.pw,
                          pooling_avg_include_padding, 1 if self.cover_all else 0)

