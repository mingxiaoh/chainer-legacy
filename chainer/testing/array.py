import numpy as np

from chainer import cuda
from chainer import utils


def assert_allclose(x, y, atol=1e-5, rtol=1e-4, verbose=True):
    """Asserts if some corresponding element of x and y differs too much.

    This function can handle both CPU and GPU arrays simultaneously.

    Args:
        x: Left-hand-side array.
        y: Right-hand-side array.
        atol (float): Absolute tolerance.
        rtol (float): Relative tolerance.
        verbose (bool): If ``True``, it outputs verbose messages on error.

    """
    x = cuda.to_cpu(utils.force_array(x))
    y = cuda.to_cpu(utils.force_array(y))
    try:
        np.testing.assert_allclose(
            x, y, atol=atol, rtol=rtol, verbose=verbose)
    except Exception:
        print('error:', np.abs(x - y).max())
        raise


def expect_allclose(act, ref, atol=1e-4, rtol=1e-4):
    """Failed if some corresponding element of act and ref differs too much.

    Args:
        act: Left-hand-side array.
        ref: Right-hand-side array.
        atol (float): Absolute tolerance.
        rtol (float): Relative tolerance.
    """
    act = cuda.to_cpu(utils.force_array(act))
    ref = cuda.to_cpu(utils.force_array(ref))
    if act.size != ref.size or act.itemsize != ref.itemsize or act.shape != ref.shape:
        return False

    total = 0
    mismatched = 0
    act_it = np.nditer(act, flags=['c_index'], op_flags=['readonly'], order='C')
    ref_it = np.nditer(ref, flags=['c_index'], op_flags=['readonly'], order='C')
    while (not act_it.finished) and (not ref_it.finished):
        total += 1
        diff = abs(act_it[0] - ref_it[0])
        if diff > (atol + rtol * abs(ref_it[0])):
            if mismatched == 0:
                print('[ __act__ , __ref__ , __diff__ , __index__ ]')
            mismatched += 1
            print(['%.8f' % act_it[0], '%.8f' % ref_it[0], '%.8f' % diff, '%d' % total])
        act_it.iternext()
        ref_it.iternext()

    if total != act.size:
        return False
    elif mismatched > 0:
        print('mismatched rate %s' % format(mismatched / total, '3.6%'))
        return False
    else:
        return True


def expect_allnear(act, ref, atol=1e-4, rtol=1e-4, ctol=0.0):
    """Failed if some corresponding element of act and ref differs too much.

    Args:
        act: Left-hand-side array.
        ref: Right-hand-side array.
        atol (float): Absolute tolerance.
        rtol (float): Relative tolerance.
        ctol (float): Compromise tolerance.
    """
    act = cuda.to_cpu(utils.force_array(act))
    ref = cuda.to_cpu(utils.force_array(ref))
    if act.size != ref.size or act.itemsize != ref.itemsize or act.shape != ref.shape:
        return False

    total = 0
    mismatched = 0
    act_it = np.nditer(act, flags=['c_index'], op_flags=['readonly'], order='C')
    ref_it = np.nditer(ref, flags=['c_index'], op_flags=['readonly'], order='C')
    while (not act_it.finished) and (not ref_it.finished):
        total += 1
        var = act_it[0] - ref_it[0]
        if abs(ref_it[0]) > atol:
            var = var / ref_it[0]
        if abs(var - ctol) >= rtol:
            if mismatched == 0:
                print('[ __act__ , __ref__ , __var__ , __index__ ]')
            mismatched += 1
            print(['%.8f' % act_it[0], '%.8f' % ref_it[0], '%.8f' % var, '%d' % total])
        act_it.iternext()
        ref_it.iternext()

    if total != act.size:
        return False
    elif mismatched > 0:
        print('mismatched rate %s' % format(mismatched / total, '3.6%'))
        return False
    else:
        return True
