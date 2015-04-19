"""Common routines to use CuDNN."""

import ctypes
import libcudnn
import numpy

def get_ptr(x):
    return ctypes.c_void_p(int(x.gpudata))

class Auto(object):
    """Object to be destoryed automatically."""
    def __init__(self, value, destroyer):
        self.value = value
        self.destroyer = destroyer

    def __del__(self):
        try:
            self.destroyer(self.value)
        except:
            pass

_default_handle = None

def get_default_handle():
    """Get the default handle of CuDNN."""
    global _default_handle
    if _default_handle is None:
        _default_handle = Auto(libcudnn.cudnnCreate(), libcudnn.cudnnDestroy)
    return _default_handle.value

_dtypes = {numpy.dtype('float32'): libcudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
           numpy.dtype('float64'): libcudnn.cudnnDataType['CUDNN_DATA_DOUBLE']}

def get_tensor_desc(x, h, w, form='CUDNN_TENSOR_NCHW'):
    """Create a tensor descriptor for given settings."""
    n = x.shape[0]
    c = x.size / (n * h * w)
    desc = libcudnn.cudnnCreateTensorDescriptor()
    libcudnn.cudnnSetTensor4dDescriptor(
        desc, libcudnn.cudnnTensorFormat[form], _dtypes[x.dtype], n, c, h, w)
    return Auto(desc, libcudnn.cudnnDestroyTensorDescriptor)

def get_conv_bias_desc(x):
    """Create a bias tensor descriptor."""
    desc = libcudnn.cudnnCreateTensorDescriptor()
    libcudnn.cudnnSetTensor4dDescriptor(
        desc, libcudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW'], _dtypes[x.dtype],
        1, x.size, 1, 1)
    return Auto(desc, libcudnn.cudnnDestroyTensorDescriptor)

_default_conv_mode = libcudnn.cudnnConvolutionMode['CUDNN_CROSS_CORRELATION']

def get_filter4d_desc(x, mode=_default_conv_mode):
    """Create a 2d convolution filter descriptor."""
    k, c, h, w = x.shape
    desc = libcudnn.cudnnCreateFilterDescriptor()
    libcudnn.cudnnSetFilter4dDescriptor(desc, _dtypes[x.dtype], k, c, h, w)
    return Auto(desc, libcudnn.cudnnDestroyFilterDescriptor)

def get_conv2d_desc(pad, stride, mode=_default_conv_mode):
    """Create a 2d convolution descriptor."""
    desc = libcudnn.cudnnCreateConvolutionDescriptor()
    libcudnn.cudnnSetConvolution2dDescriptor(
        desc, pad[0], pad[1], stride[0], stride[1], 1, 1, mode)
    return Auto(desc, libcudnn.cudnnDestroyConvolutionDescriptor)
