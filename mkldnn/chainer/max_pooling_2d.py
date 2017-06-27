from mkldnn.chainer.pooling_2d import Pooling2DMKLDNN, Pooling2DForward, Pooling2DBackward
from mkldnn.api.support import pooling_max


class MaxPooling2DMKLDNN(Pooling2DMKLDNN):

    """Max pooling over a set of 2d planes."""

    def forward_cpu(self, x):
        cc = Pooling2DForward(x, pooling_max, ksize=(self.kh, self.kw),
                              stride=(self.sy, self.sx),
                              pad=(self.ph, self.pw), cover_all=self.cover_all,
                              pos=(self.rank, self.fanout))

        self.hint = cc.hint
        self.ws = cc.ws
        y, = cc.execute_on()
        return y,

    def backward_cpu(self, x, gy):
        cc = Pooling2DBackward(x, gy[0], self.hint, self.ws, pooling_max,
                               ksize=(self.kh, self.kw),
                               stride=(self.sy, self.sx),
                               pad=(self.ph, self.pw), cover_all=self.cover_all,
                               pos=(self.rank, self.fanout))
        gx, = cc.execute_on()
        return gx,
