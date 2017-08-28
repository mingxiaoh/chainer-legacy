import nose

from chainer.testing import array  # NOQA
from chainer.testing import helper  # NOQA
from chainer.testing import parameterized  # NOQA
from chainer.testing import unary_math_function_test  # NOQA


from chainer.testing.array import assert_allclose  # NOQA
from chainer.testing.array import expect_allclose # NOQA
from chainer.testing.helper import with_requires  # NOQA
from chainer.testing.parameterized import parameterize  # NOQA
from chainer.testing.parameterized import product  # NOQA
from chainer.testing.parameterized import product_dict  # NOQA
from chainer.testing.unary_math_function_test import unary_math_function_unittest  # NOQA

import chainer.variable


def run_module(name, file):
    """Run current test cases of the file.

    Args:
        name: __name__ attribute of the file.
        file: __file__ attribute of the file.
    """
    chainer.variable.RANK_START = 1

    if name == '__main__':

        nose.runmodule(argv=[file, '-vvs', '-x', '--ipdb', '--ipdb-failure'],
                       exit=False)
