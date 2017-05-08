import warnings
import numpy
import os

avaiable = False
mkldnn_enabled = False

try:
    import mkldpy
    from mkldpy import mkldnn
    avaiable = True
except Exception as e:
    _resolution_error = e
    warnings.warn(str(_resolution_error))
    warnings.warn("WARNING: mkldnn enviorment is not corretly set up \n see(https://github.com/intel/chainer#installation).")

if avaiable:
    _mkldnn_disabled_by_user = int(os.environ.get('CHAINER_MKLDNN', '1')) == 0
    mkldnn_enabled = not _mkldnn_disabled_by_user

if avaiable and not mkldnn_enabled:
    warnings.warn("WARNING: mkldnn acceleration is disabled!!")

"""
MKLDNN backend switch
"""
enable_conv = True
enable_max_pooling = True
enable_avg_pooling = True
enable_lrn = True
enable_relu = True
enable_softmax = False
enable_linear = True
enable_softmax_cross_entropy = False
enable_concat = True
enable_acc_grad = True
enable_batch_normalization = True
cosim_enabled = False

supportTypes = (numpy.float32,)

def SupportedInput(tul):
    isSupportType = True
    for x in tul:
        if len(x) == 0:
            continue
        if x[0].dtype not in supportTypes:
            isSupportType = False
            break
        else:
            isSupportType = True
    return isSupportType

def set_mkldnn_enabled():
    global mkldnn_enabled
    mkldnn_enabled = True

def set_mkldnn_disabled():
    global mkldnn_enabled
    mkldnn_enabled = False

def enable_cosim():
    global mkldnn_enabled
    global cosim_enabled
    return mkldnn_enabled and cosim_enabled

def enable_convF(tul):
        return mkldnn_enabled and SupportedInput(tul) and enable_conv


def enable_max_poolingF(tul):
        return mkldnn_enabled and SupportedInput(tul) and enable_max_pooling


def enable_avg_poolingF(tul):
        return mkldnn_enabled and SupportedInput(tul) and enable_avg_pooling


def enable_lrnF(tul):
        return mkldnn_enabled and SupportedInput(tul) and enable_lrn


def enable_reluF(tul):
        return mkldnn_enabled and SupportedInput(tul) and enable_relu


def enable_softmaxF(tul):
        return mkldnn_enabled and SupportedInput(tul) and enable_softmax


def enable_linearF(tul):
        return mkldnn_enabled and SupportedInput(tul) and enable_linear


def enable_softmax_cross_entropyF(tul):
        return mkldnn_enabled and SupportedInput(tul) and enable_softmax_cross_entropy


def enable_concatF(tul):
        return mkldnn_enabled and SupportedInput(tul) and enable_concat


def enable_acc_gradF(tul):
        return mkldnn_enabled and SupportedInput(tul) and enable_acc_grad


def enable_batch_normalizationF(tul):
        return mkldnn_enabled and SupportedInput(tul) and enable_batch_normalization
