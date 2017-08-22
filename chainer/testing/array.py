import numpy as np
import mkldnn.api.cosim_dump as cdump

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


def expect_allclose(act, ref, atol=1e-4, rtol=1e-4, verbose=True):
    """Failed if some corresponding element of act and ref differs too much.

    Args:
        act: Left-hand-side array.
        ref: Right-hand-side array.
        atol (float): Absolute tolerance.
        rtol (float): Relative tolerance.
        verbose (bool): If ``True``, it outputs verbose messages on error.
    """
    act = cuda.to_cpu(utils.force_array(act))
    ref = cuda.to_cpu(utils.force_array(ref))
    if act.size != ref.size or act.itemsize != ref.itemsize or act.shape != ref.shape:
        return False

    act = np.ascontiguousarray(act)
    ref = np.ascontiguousarray(ref)

    cc = cdump.cosim_check()
    cc.set_act_view(act)
    cc.set_ref_view(ref)
    return cc.expect_allclose(atol, rtol)


