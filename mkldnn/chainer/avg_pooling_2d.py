from mkldnn.chainer.pooling_2d import Pooling2DMKLDNN
from mkldnn.chainer.pooling_2d import Pooling2DForward
from mkldnn.chainer.pooling_2d import Pooling2DBackward
from mkldnn.api.support import pooling_avg_include_padding


class AvgPooling2DMKLDNN(Pooling2DMKLDNN):

    """Average pooling over a set of 2d planes."""

    def forward_cpu(self, x):
        cc = Pooling2DForward(x, pooling_avg_include_padding, ksize=(self.kh, self.kw),
                              stride=(self.sy, self.sx),
                              pad=(self.ph, self.pw), cover_all=self.cover_all,
                              pos=(self.rank, self.fanout))

        self.hint = cc.hint
        self.ws = cc.ws
        y, = cc.execute_on()
        return y,

    def backward_cpu(self, x, gy):
        cc = Pooling2DBackward(x, gy[0], self.hint, self.ws, pooling_avg_include_padding,
                               ksize=(self.kh, self.kw),
                               stride=(self.sy, self.sx),
                               pad=(self.ph, self.pw), cover_all=self.cover_all,
                               pos=(self.rank, self.fanout))
        gx, = cc.execute_on()
        return gx,
