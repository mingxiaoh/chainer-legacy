available = False

try:
    import mkldnn
    from mkldnn.mdarray import mdarray
    from mkldnn import chainer
    from mkldnn.chainer import basic_math
    from mkldnn.chainer import fanout
    from mkldnn.chainer import runtime
    from mkldnn.chainer import sum
    # Modules listed depend on chainer.
    from mkldnn.chainer import avg_pooling_2d
    from mkldnn.chainer import bn
    from mkldnn.chainer import concat
    from mkldnn.chainer import convolution_2d
    from mkldnn.chainer import linear
    from mkldnn.chainer import lrn
    from mkldnn.chainer import max_pooling_2d
    from mkldnn.chainer import pooling_2d
    from mkldnn.chainer import relu

    available = True
except Exception as ex:
    error_info = ex

    class mdarray(object):
        pass
