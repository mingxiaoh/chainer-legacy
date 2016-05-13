from chainer.functions.array import concat
from chainer.functions.pooling import max_pooling_2d
from chainer import link
from chainer.links.connection import convolution_2d
from chainer.links.connection import linear


class Sequential(link.ChainList):

    def __call__(self, x):
        for l in self.links:
            x = l(x)
        return x


class Inception(link.ChainList):

    def __call__(self, x):
        xs = [l(x) for l in self.links]
        return concat.concat(xs)


class AuxiliaryClassifier(link.Chain):
    pass


class InceptionV3BN(link.Chain):
    """

    https://github.com/tensorflow/models/blob/master/inception/inception/slim/inception_model.py
    """

    def 

    def __init__(self, pool_type):
        convolutions = Sequential(
            convolution_2d.Convolution2D(3, 32, 3, 2),
            convolution_2d.Convolution2D(32, 32, 3),
            convolution_2d.Convolution2D(32, 64, 3, 1, 2),
            convolution_2d.Convolution2D(64, 80, 3, 1),
            convolution_2d.Convolution2D(80, 192, 3, 2),
            convolution_2d.Convolution2D(192, 288, 3, 1, 2))
        inception1 = Sequential(
            Inception(
                Sequential(
                    convolution_2d.Convolution2D(),
                    convolution_2d.Convolution2D()),
                Sequential(
                    convolution_2d.Convolution2D(),
                    convolution_2d.Convolution2D()),
                Sequential(
                    convolution_2d.Convolution2D(),
                    pool=max_pooling_2d(),
                )
                Sequential(
                    convolution_2d.Convolution2D())
            ) for _ in six.moves.range(3))
        grid_reduction1 = Inception(
            Sequential(
                convolution_2d.Convolution2D(),
                convolution_2d.Convolution2D()),
                convolution_2d.Convolution2D())),
            Sequential(
                convolution_2d.Convolution2D(),
                convolution_2d.Convolution2D())),
            Sequential(
                Identity(),
                pool=max_pooling_2d()))
        inception2 = Sequential(
            Inception(
                Sequential(
                    convolution_2d.Convolution2D(),
                    convolution_2d.Convolution2D(),
                    convolution_2d.Convolution2D(),
                    convolution_2d.Convolution2D(),
                    convolution_2d.Convolution2D()),
                Sequential(
                    convolution_2d.Convolution2D(),
                    convolution_2d.Convolution2D(),
                    convolution_2d.Convolution2D()),
                Sequential(
                    convolution_2d.Convolution2D(),
                    pool=max_pooling_2d(),
                )
                Sequential(
                    convolution_2d.Convolution2D())
            ) for _ in six.moves.range(5))
        grid_reduction2 = Inception(
            Sequential(
                convolution_2d.Convolution2D(),
                convolution_2d.Convolution2D()),
                convolution_2d.Convolution2D())),
            Sequential(
                convolution_2d.Convolution2D(),
                convolution_2d.Convolution2D())),
            Sequential(
                Identity(),
                pool=max_pooling_2d()))
        inception2 = Sequential(
            Inception(
                Sequential(
                    convolution_2d.Convolution2D(),
                    convolution_2d.Convolution2D(),
                    Inception(
                        convolution_2d.Convolution2D(),
                        convolution_2d.Convolution2D()
                    )),
                Sequential(
                    convolution_2d.Convolution2D(),
                    Inception(
                        convolution_2d.Convolution2D(),
                        convolution_2d.Convolution2D()
                    )
                    convolution_2d.Convolution2D()),
                Sequential(
                    convolution_2d.Convolution2D(),
                    pool=max_pooling_2d(),
                )
                Sequential(
                    convolution_2d.Convolution2D())
            ) for _ in six.moves.range(2))

        auxiliary_classifier = Sequential(
            
        )

        super(InceptionV3BN, self).__init__(
            convolutions=convolutions,
            inceptions=ChainList(inception1, inception2, inception3)
            grid_reduction=ChainList(grid_reduction1, grid_reduction2),
            auxiliary_classifier=AuxliaryClassifier(),
            grid_reduction2=Inception()
            inception3=ChainList(
                Inception(),
                Inception(),
                Inception()
            )
            linear=linear.Linear(2048, 1000))
            self.pool_type=pool_type

    def __call__(self, x):
        """Computes the output of the module

        Args:
            x(~chainer.Variable): Input variable.
        """

        
