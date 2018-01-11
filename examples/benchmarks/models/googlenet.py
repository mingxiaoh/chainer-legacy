import chainer
import chainer.functions as F
import chainer.links as L


class GoogLeNet(chainer.Chain):
    def __init__(self, insize=224):
        super(GoogLeNet, self).__init__(
            conv1=L.Convolution2D(3,  64, 7, stride=2, pad=3),
            conv2_reduce=L.Convolution2D(64,  64, 1),
            conv2=L.Convolution2D(64, 192, 3, stride=1, pad=1),
            inc3a=L.Inception(192,  64,  96, 128, 16,  32,  32),
            inc3b=L.Inception(256, 128, 128, 192, 32,  96,  64),
            inc4a=L.Inception(480, 192,  96, 208, 16,  48,  64),
            inc4b=L.Inception(512, 160, 112, 224, 24,  64,  64),
            inc4c=L.Inception(512, 128, 128, 256, 24,  64,  64),
            inc4d=L.Inception(512, 112, 144, 288, 32,  64,  64),
            inc4e=L.Inception(528, 256, 160, 320, 32, 128, 128),
            inc5a=L.Inception(832, 256, 160, 320, 32, 128, 128),
            inc5b=L.Inception(832, 384, 192, 384, 48, 128, 128),
            loss3_fc=L.Linear(1024, 1000),

            loss1_conv=L.Convolution2D(512, 128, 1),
            loss1_fc1=L.Linear(4 * 4 * 128, 1024),
            loss1_fc2=L.Linear(1024, 1000),

            loss2_conv=L.Convolution2D(528, 128, 1),
            loss2_fc1=L.Linear(4 * 4 * 128, 1024),
            loss2_fc2=L.Linear(1024, 1000)
        )
        self.insize = insize

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.local_response_normalization(
            F.max_pooling_2d(h, 3, stride=2), n=5)

        h = F.relu(self.conv2_reduce(h))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(
            F.local_response_normalization(h, n=5), 3, stride=2)

        h = self.inc3a(h)
        h = self.inc3b(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.inc4a(h)

        if chainer.config.train:
            out1 = F.average_pooling_2d(h, 5 * self.insize // 224,
                                        stride=3 * self.insize // 224)
            out1 = F.relu(self.loss1_conv(out1))
            out1 = F.relu(self.loss1_fc1(out1))
            out1 = self.loss1_fc2(out1)

        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        if chainer.config.train:
            out2 = F.average_pooling_2d(h, 5 * self.insize // 224,
                                        stride=3 * self.insize // 224)
            out2 = F.relu(self.loss2_conv(out2))
            out2 = F.relu(self.loss2_fc1(out2))
            out2 = self.loss2_fc2(out2)

        h = self.inc4e(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.inc5a(h)
        h = self.inc5b(h)

        h = F.dropout(F.average_pooling_2d(h, 7 * self.insize // 224,
                                           stride=1 * self.insize // 224), 0.4)
        out3 = self.loss3_fc(h)
        return out1, out2, out3
