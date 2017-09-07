import numpy as np
import mkldnn.api.cosim_dump as cdump

from functools import reduce
from chainer import function
from chainer.utils import force_array, type_check

from mkldnn.chainer import is_cosim, is_cosim_continue, plain_array


class Dropout(function.Function):
    """
    """

    def __init__(self, dropout_ratio, mask):
        self.dropout_ratio = dropout_ratio
        self.mask = mask

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        return self.mask * x[0],

    def backward(self, x, gy):
        return self.mask * gy[0],


def expect_allclose(act, ref, atol=1e-4, rtol=1e-4, verbose=True):
    """Failed if some corresponding element of act and ref differs too much.

    Args:
        act: Left-hand-side array.
        ref: Right-hand-side array.
        atol (float): Absolute tolerance.
        rtol (float): Relative tolerance.
        verbose (bool): If ``True``, it outputs verbose messages on error.
    """
    if not isinstance(act, np.ndarray) or not isinstance(ref, np.ndarray):
        print('\tWARNING: wrong array types!')
        return False

    act = force_array(act)
    ref = force_array(ref)
    if act.size != ref.size or act.itemsize != ref.itemsize or act.shape != ref.shape:
        print('\tWARNING: size is not matched!\nsize: act=%d ref=%d itemsize: act=%d ref=%d'
              % (act.size, ref.size, act.itemsize, ref.itemsize),
              ' shape: act=', act.shape, ' ref=', ref.shape,
              ' dtype: act=', act.dtype, ' ref=', ref.dtype)
        return False

    act = np.ascontiguousarray(act)
    ref = np.ascontiguousarray(ref)

    cc = cdump.cosim_check()
    cc.set_act_view(act)
    cc.set_ref_view(ref)
    return cc.expect_allclose(atol, rtol)


def verify_results(func, acts, refs, inputs, out_grads=None):
    """
    """
    print('\tCosim verify results for %s <rank=%d, fanout=%d>'
          % (func.__class__.__name__, func.rank, func.fanout))
    check_options = {'atol': 1e-2, 'rtol': 1e-2, 'verbose': True}

    if acts is None and refs is None:
        print('\tWARNING: input results are None!')
        return True
    elif acts is None or refs is None:
        if is_cosim_continue():
            print('\tWARNING: cosim, input results are None!')
            return False
        else:
            raise KeyError('Cosim, input results are None!')

    size = len(acts)
    if size != len(refs):
        if is_cosim_continue():
            print('\tWARNING: cosim, lengths of results are different'
                  + ' <acts_size=%d refs_size=%d>!' % (size, len(refs)))
            return False
        else:
            raise KeyError('Cosim, lengths of results are different'
                           + ' <acts_size=%d refs_size=%d>!' % (size, len(refs)))

    ret = True
    for i in range(size):
        if acts[i] is None and refs[i] is None:
            continue
        elif acts[i] is None or refs[i] is None:
            ret = False
            if is_cosim_continue():
                print('\tWARNING: cosim, one input result is None!')
                continue
            else:
                raise KeyError('Cosim, one  input result is None!')

        if not expect_allclose(*plain_array((acts[i], refs[i])), **check_options):
            ret = False
            if is_cosim_continue():
                print('\tCosim, mismatched in %s #%d result!' % (func.__class__.__name__, i))
                continue
            else:
                if hasattr(func, 'dump_to_file'):
                    print('\tDump input parameters to file for %s' % func.__class__.__name__)
                    func.dump_to_file(inputs, out_grads)
                raise KeyError('Cosim, mismatched in #%d result of %s!'
                               % (i, func.__class__.__name__))

    return ret


def cosim_verify(func, acts, inputs, out_grads=None):
    """
    """
    if not is_cosim() or not hasattr(func, 'cosim_func'):
        return

    from mkldnn.chainer.bn import BnMKLDNN
    from mkldnn.chainer.concat import ConcatMKLDNN
    from mkldnn.chainer.linear import LinearFunctionMKLDNN

    if out_grads is None:
        print('\tFORWARD cosim for %s <rank=%d, fanout=%d>'
              % (func.__class__.__name__, func.rank, func.fanout))

        # Reshape W for Linear
        orig_shape = None
        if isinstance(func, LinearFunctionMKLDNN):
            assert len(inputs) >= 2
            if inputs[1].ndim == 4:
                inputs = plain_array(inputs)
                orig_shape = inputs[1].shape
                W = np.reshape(inputs[1], (inputs[1].shape[0],
                               reduce(lambda x, y: x * y, inputs[1].shape[1:])))
                inputs = tuple([x if i != 1 else W for i, x in enumerate(inputs)])

        refs = plain_array((func.cosim_func(*plain_array(inputs)), ))

        # Restore W for Linear
        if isinstance(func, LinearFunctionMKLDNN) and orig_shape is not None:
            assert len(inputs) >= 2
            if inputs[1].ndim == 2:
                inputs = plain_array(inputs)
                W = np.reshape(inputs[1], orig_shape)
                inputs = tuple([x if i != 1 else W for i, x in enumerate(inputs)])
                orig_shape = None

        # Reshape y for Concat
        elif isinstance(func, ConcatMKLDNN):
            if refs[0].ndim == 3 and acts[0].ndim == 4 and \
                    acts[0].shape[0] == 1 and \
                    refs[0].shape[0] == acts[0].shape[1] and \
                    refs[0].shape[1] == acts[0].shape[2] and \
                    refs[0].shape[2] == acts[0].shape[3]:
                ref0 = np.reshape(refs[0], acts[0].shape)
                refs = tuple([x if i != 0 else ref0 for i, x in enumerate(refs)])

        elif isinstance(func, BnMKLDNN):
            if refs[0].ndim == 4 and acts[0].ndim == 2 and \
                    refs[0].size == acts[0].size and \
                    refs[0].itemsize == acts[0].itemsize and \
                    refs[0].shape[0] == acts[0].shape[0] and \
                    acts[0].shape[1] == reduce(lambda x, y: x * y, refs[0].shape[1:]):
                ref0 = np.reshape(refs[0], acts[0].shape)
                refs = tuple([x if i != 0 else ref0 for i, x in enumerate(refs)])

        if not verify_results(func, acts, refs, inputs):
            print('\tFailed in FORWARD cosim for %s <rank=%d, fanout=%d>'
                  % (func.__class__.__name__, func.rank, func.fanout))

    else:
        print('\tBACKWARD cosim for %s <rank=%d, fanout=%d>'
              % (func.__class__.__name__, func.rank, func.fanout))
        refs = plain_array(func.cosim_func.backward(plain_array(inputs), plain_array(out_grads)))

        # Reshape gW for Linear
        if isinstance(func, LinearFunctionMKLDNN):
            assert len(acts) >= 2 and len(refs) >= 2
            if acts[1].ndim == 4 and refs[1].ndim == 2:
                ref1 = np.reshape(refs[1], acts[1].shape)
                refs = tuple([x if i != 1 else ref1 for i, x in enumerate(refs)])

        elif isinstance(func, BnMKLDNN):
            if refs[0].ndim == 4 and acts[0].ndim == 2 and \
                    refs[0].size == acts[0].size and \
                    refs[0].itemsize == acts[0].itemsize and \
                    refs[0].shape[0] == acts[0].shape[0] and \
                    acts[0].shape[1] == reduce(lambda x, y: x * y, refs[0].shape[1:]):
                ref0 = np.reshape(refs[0], acts[0].shape)
                refs = tuple([x if i != 0 else ref0 for i, x in enumerate(refs)])

        if not verify_results(func, acts, refs, inputs, out_grads):
            print('\tFailed in BACKWARD cosim for %s <rank=%d, fanout=%d>'
                  % (func.__class__.__name__, func.rank, func.fanout))


