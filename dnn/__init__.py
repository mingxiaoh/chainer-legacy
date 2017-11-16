import os

from chainer.configuration import config  # NOQA
from chainer.configuration import global_config  # NOQA


global_config.cosim = bool(int(os.environ.get('CHAINER_ENABLE_COSIM', '0')))


def is_cosim():
    """Get the cosim mode.

    Returns:
        bool: Return ``True`` if chainer is in cosim mode.
    """
    return config.cosim
