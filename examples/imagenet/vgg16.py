#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import initializers
import chainer.links as L
import chainer.functions as F


class VGG16(chainer.Chain):

    """
    VGGNet
    - It takes (224, 224, 3) sized image as imput
    """

    insize = 224

    def __init__(self):
        initialW = initializers.HeNormal(1.0)
        # initialW = initializers.GlorotUniform(scale=1.0) # Xavier

        super(VGG16, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, initialW=initialW,
                                    stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, initialW=initialW,
                                    stride=1, pad=1),

            conv2_1=L.Convolution2D(64, 128, 3, initialW=initialW,
                                    stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, initialW=initialW,
                                    stride=1, pad=1),

            conv3_1=L.Convolution2D(128, 256, 3, initialW=initialW,
                                    stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, initialW=initialW,
                                    stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, initialW=initialW,
                                    stride=1, pad=1),

            conv4_1=L.Convolution2D(256, 512, 3, initialW=initialW,
                                    stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, initialW=initialW,
                                    stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, initialW=initialW,
                                    stride=1, pad=1),

            conv5_1=L.Convolution2D(512, 512, 3, initialW=initialW,
                                    stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, initialW=initialW,
                                    stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, initialW=initialW,
                                    stride=1, pad=1),

            fc6=L.Linear(25088, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 1000)
        )
        self.train = True

    def __call__(self, x, t):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.dropout(F.relu(self.fc6(h)), ratio=0.5)
        h = F.dropout(F.relu(self.fc7(h)), ratio=0.5)
        h = self.fc8(h)

        if self.train:
            loss = F.softmax_cross_entropy(h, t)
            chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
            return loss
        else:
            self.pred = F.softmax(h)
            return self.pred
