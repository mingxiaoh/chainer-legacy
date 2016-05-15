import six

from chainer.functions.array import concat
from chainer.functions.pooling import max_pooling_2d
from chainer.functions.pooling import average_pooling_2d
from chainer import link
from chainer.links.connection import convolution_2d
from chainer.links.connection import linear
from chainer.links.normalization import batch_normalization
from chainer.functions.noise import dropout


class AuxConv(link.Chain):

    def __init__(self, conv, batch_norm=True, pool=None):
        super(AuxConv, self).__init__(conv=conv)
        if batch_norm:
            out_channel = conv.W.data.shape[0]
            self.add_link('batch_norm',
                          batch_normalization.BatchNormalization(out_channel))
        self.pool = pool

    def __call__(self, x, train=True):
        if self.pool:
            x = self.pool(x)
        x = self.conv(x)
        return self.batch_norm(x, test=not train)


class Sequential(link.ChainList):

    def __call__(self, x, train=True):
        for l in self:
            x = l(x, train)
        return x


class Inception(link.ChainList):

    def __init__(self, *links, **kw):
        super(Inception, self).__init__(*links)
        self.pool = kw.get('pool', None)

    def __call__(self, x, train=True):
        xs = [l(x, train) for l in self]
        if self.pool:
            xs.append(self.pool(x))
        return concat.concat(xs)


class InceptionV3(link.Chain):
    """Inception V3

    http://arxiv.org/abs/1512.00567
    https://github.com/tensorflow/models/blob/master/inception/inception/slim/inception_model.py
    """

    def __init__(self):
        convolution = link.ChainList(
            AuxConv(convolution_2d.Convolution2D(3, 32, 3, 2)),
            AuxConv(convolution_2d.Convolution2D(32, 32, 3)),
            AuxConv(convolution_2d.Convolution2D(32, 64, 3, 1, 1)),
            AuxConv(convolution_2d.Convolution2D(64, 80, 3, 1, 1)),
            AuxConv(convolution_2d.Convolution2D(80, 192, 3)))

        inception0 = Sequential(
            *[Inception(
                    # 1x1
                    Sequential(
                        AuxConv(convolution_2d.Convolution2D(input_channel, 64, 1))),
                    # 5x5
                    Sequential(
                        AuxConv(convolution_2d.Convolution2D(input_channel, 48, 1)),
                        AuxConv(convolution_2d.Convolution2D(48, 64, 5, pad=2))),
                    # double 3x3
                    Sequential(
                        AuxConv(convolution_2d.Convolution2D(input_channel, 64, 1)),
                        AuxConv(convolution_2d.Convolution2D(64, 96, 3, pad=1)),
                        AuxConv(convolution_2d.Convolution2D(96, 96, 3, pad=1))),
                    # pool
                    Sequential(
                        AuxConv(convolution_2d.Convolution2D(input_channel, pool_channel, 3, pad=1),
                                pool=max_pooling_2d.MaxPooling2D(3, stride=1, pad=1))))
              for input_channel, pool_channel in zip([192, 256, 288], [32, 64, 64])])

        grid_reduction0 = Inception(
            # strided 3x3
            Sequential(
                AuxConv(convolution_2d.Convolution2D(288, 384, 3, 2))),
            # double 3x3
            Sequential(
                AuxConv(convolution_2d.Convolution2D(288, 64, 1)),
                AuxConv(convolution_2d.Convolution2D(64, 96, 3, pad=1)),
                AuxConv(convolution_2d.Convolution2D(96, 96, 3, 2))),
            # pool
            pool=max_pooling_2d.MaxPooling2D(3, 2))

        inception1 = Sequential(
            *[Inception(
                # 1x1
                Sequential(
                    AuxConv(convolution_2d.Convolution2D(768, 192, 1))),
                # 7x7
                Sequential(
                    AuxConv(convolution_2d.Convolution2D(768, c, 1)),
                    AuxConv(convolution_2d.Convolution2D(c, c, (1, 7), pad=(0, 3))),
                    AuxConv(convolution_2d.Convolution2D(c, 192, (7, 1), pad=(3, 0)))),
                # double 7x7
                Sequential(
                    AuxConv(convolution_2d.Convolution2D(768, c, 1)),
                    AuxConv(convolution_2d.Convolution2D(c, c, (1, 7), pad=(0, 3))),
                    AuxConv(convolution_2d.Convolution2D(c, c, (7, 1), pad=(3, 0))),
                    AuxConv(convolution_2d.Convolution2D(c, c, (1, 7), pad=(0, 3))),
                    AuxConv(convolution_2d.Convolution2D(c, 192, (7, 1), pad=(3, 0)))),
                # pool
                Sequential(
                    AuxConv(convolution_2d.Convolution2D(768, 192, 3, pad=1),
                            pool=average_pooling_2d.AveragePooling2D(3, stride=1, pad=1))),
            ) for c in [128, 160, 160, 192]])

        grid_reduction1 = Inception(
            # strided 3x3
            Sequential(
                AuxConv(convolution_2d.Convolution2D(768, 192, 1)),
                AuxConv(convolution_2d.Convolution2D(192, 320, 3, 2))),
            # 7x7 and 3x3
            Sequential(
                AuxConv(convolution_2d.Convolution2D(768, 192, 1)),
                AuxConv(convolution_2d.Convolution2D(192, 192, (1, 7), pad=(0, 3))),
                AuxConv(convolution_2d.Convolution2D(192, 192, (7, 1), pad=(3, 0))),
                AuxConv(convolution_2d.Convolution2D(192, 192, 3, 2))),
            # pool
            pool=max_pooling_2d.MaxPooling2D(3, 2))

        inception2 = Sequential(
            *[Inception(
                # 1x1
                Sequential(
                    AuxConv(convolution_2d.Convolution2D(input_channel, 320, 1))),
                # 3x3
                Sequential(
                    AuxConv(convolution_2d.Convolution2D(input_channel, 384, 1)),
                    Inception(
                        AuxConv(convolution_2d.Convolution2D(384, 384, (1, 3), pad=(0, 1))),
                        AuxConv(convolution_2d.Convolution2D(384, 384, (3, 1), pad=(1, 0))))),
                # double 3x3
                Sequential(
                    AuxConv(convolution_2d.Convolution2D(input_channel, 448, 1)),
                    AuxConv(convolution_2d.Convolution2D(448, 384, 3, pad=1)),
                    Inception(
                        AuxConv(convolution_2d.Convolution2D(384, 384, (1, 3), pad=(0, 1))),
                        AuxConv(convolution_2d.Convolution2D(384, 384, (3, 1), pad=(1, 0))))),
                # pool
                AuxConv(convolution_2d.Convolution2D(input_channel, 192, 3, pad=1),
                        pool=average_pooling_2d.AveragePooling2D(3, stride=1, pad=1)))
             for input_channel in [1280, 2048]])

        auxiliary_convolution = Sequential(
            AuxConv(convolution_2d.Convolution2D(768, 128, 1),
                    pool=average_pooling_2d.AveragePooling2D(5, stride=3)),
            AuxConv(convolution_2d.Convolution2D(128, 768, 5)))

        super(InceptionV3, self).__init__(
            convolution=convolution,
            inception=link.ChainList(inception0, inception1, inception2),
            grid_reduction=link.ChainList(grid_reduction0, grid_reduction1),
            auxiliary_convolution = auxiliary_convolution,
            auxiliary_linear = linear.Linear(768, 1000),
            linear=linear.Linear(2048, 1000))

    def _convolution(self, x, train):
        x = self.convolution[0](x)
        x = self.convolution[1](x)
        x = self.convolution[2](x)
        x = max_pooling_2d.max_pooling_2d(x, 3, 2)
        x = self.convolution[3](x)
        x = self.convolution[4](x)
        x = max_pooling_2d.max_pooling_2d(x, 3, 2)
        return x

    # Original paper and TensorFlow implementation has different
    # auxiliary classifier.
    # We implement latter one.
    def _auxiliary_classifier(self, x, train):
        x = self.auxiliary_convolution(x, train)
        return self.auxiliary_linear(x)

    def _classifier(self, x, train):
        x = average_pooling_2d.average_pooling_2d(x, 8)
        x = dropout.dropout(x, train=train)
        x = self.linear(x)
        return x

    def __call__(self, x, train=True):
        """Computes the output of the module

        Args:
            x(~chainer.Variable): Input variable.
        """

        x = self._convolution(x, train)
        assert x.data.shape[1:] == (192, 35, 35), 'actual={}'.format(x.data.shape[1:])
        x = self.inception[0](x, train)
        assert x.data.shape[1:] == (288, 35, 35), 'actual={}'.format(x.data.shape[1:])
        x = self.grid_reduction[0](x, train)
        assert x.data.shape[1:] == (768, 17, 17), 'actual={}'.format(x.data.shape[1:])
        x = self.inception[1](x, train)
        assert x.data.shape[1:] == (768, 17, 17), 'actual={}'.format(x.data.shape[1:])
        y_aux = self._auxiliary_classifier(x, train)
        x = self.grid_reduction[1](x, train)
        assert x.data.shape[1:] == (1280, 8, 8), 'actual={}'.format(x.data.shape[1:])
        x = self.inception[2](x, train)
        assert x.data.shape[1:] == (2048, 8, 8), 'actual={}'.format(x.data.shape[1:])
        y = self._classifier(x, train)
