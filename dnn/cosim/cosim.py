import logging
import numpy as np
import os

from chainer.configuration import config  # NOQA
from chainer.configuration import global_config  # NOQA
from chainer import variable  # NOQA
from chainer.utils import force_array  # NOQA

from dnn._dnn import mdarray

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s]: %(message)s')
global_config.cosim = bool(int(os.environ.get('CHAINER_ENABLE_COSIM', '0')))


def is_cosim():
    """Get the cosim mode.

    Returns:
        bool: Return ``True`` if chainer is in cosim mode.
    """
    return config.cosim


def plain_array(params):
    assert isinstance(params, tuple) \
           or isinstance(params, list) \
           or isinstance(params, mdarray) \
           or isinstance(params, np.ndarray) \
           or isinstance(params, variable.Variable)

    _params = ()

    if isinstance(params, variable.Variable):
        return np.array(params.data),
    elif isinstance(params, np.ndarray):
        return params,
    elif isinstance(params, mdarray):
        return np.array(params),

    for p in params:
        if isinstance(p, variable.Variable):
            p = np.array(p.data)
        if isinstance(p, mdarray):
            _params += (np.array(p),)
        else:
            _params += (p,)

    return _params


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
        logging.warning('wrong array types')
        return False

    act = force_array(act)
    ref = force_array(ref)

    if act.size != ref.size or act.itemsize != ref.itemsize or act.shape != ref.shape:
        logging.warning('size is not matched!\nsize: act={0} ref={1} itemsize: act={2} ref={3}\n'
                        'shape: act={4}, ref={5} dtype: act={6} ref={7}'
                        .format(act.size, ref.size, act.itemsize, ref.itemsize,
                                act.shape, ref.shape, act.dtype, ref.dtype))
        return False

    act = np.ascontiguousarray(act)
    ref = np.ascontiguousarray(ref)

    try:
        np.testing.assert_allclose(act, ref, rtol, atol, verbose=verbose)
    except Exception:
        return False

    return True


def verify_results(func, acts, refs, inputs):
    if acts is None and refs is None:
        logging.warning('input results are None!')
        return True
    elif acts is None or refs is None:
        logging.error('cosim: input results are None!')
        return False

    if len(acts) != len(refs):
        logging.error('cosim: lengths of results are different <acts_size={0} refs_size={1}>!'
                      .format(len(acts), len(refs)))
        return False

    check_options = {'atol': 1e-3, 'rtol': 1e-2, 'verbose': True}

    for (i, (act, ref)) in enumerate(zip(acts, refs)):
        if ref is None and act is None:
            continue
        elif ref is None or act is None:
            logging.error('cosim: one input result is None!')
            return False

        if not expect_allclose(*plain_array((act, ref)), **check_options):
            logging.error('cosim: mismatched in {0} #{1} result!\nsize: {2}, itemsize: {3}\n'
                          'shape: {4}, dtype: {5}'.format(func.__class__.__name__, i, act.size, act.itemsize,
                                                          act.shape, act.dtype))
            return False

    return True


def cosim_verify(func, acts, inputs):
    if not is_cosim():
        return

    logging.info('cosim test for function {0} ...'.format(func.__class__.__name__))

    refs = plain_array(func.forward_cpu(plain_array(inputs)))

    if not verify_results(func, acts, refs, inputs):
        logging.error('cosim test for function {0} ...FAILED'.format(func.__class__.__name__))
        raise RuntimeError

    logging.info('cosim test for function {0} ...PASS'.format(func.__class__.__name__))
