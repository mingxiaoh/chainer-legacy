import unittest

import chainer
from chainer import testing
from chainer.testing import attr

class TestUseMklDNN(unittest.TestCase):
    def test_invalid_level(self):
        # print (chainer.should_use_mkldnn('==always'))
        self.assertRaises(ValueError, chainer.should_use_mkldnn, '==auto')
    def test_invalid_config(self):
        with chainer.using_config('use_mkldnn', True):
            self.assertRaises(ValueError, chainer.should_use_mkldnn, '>=auto')
        with chainer.using_config('use_mkldnn', False):
            self.assertRaises(ValueError, chainer.should_use_mkldnn, '>=auto')
        with chainer.using_config('use_mkldnn', 'on'):
            self.assertRaises(ValueError, chainer.should_use_mkldnn, '>=auto')
    def test_valid_case_combination1(self):
        with chainer.using_config('use_mkldnn', 'always'):
            self.assertTrue(chainer.should_use_mkldnn('==always'))
            self.assertTrue(chainer.should_use_mkldnn('>=auto'))
    def test_valid_case_combination2(self):
        with chainer.using_config('use_mkldnn', 'auto'):
            self.assertFalse(chainer.should_use_mkldnn('==always'))
            self.assertTrue(chainer.should_use_mkldnn('>=auto'))
    def test_valid_case_combination3(self):
        with chainer.using_config('use_mkldnn', 'never'):
            self.assertFalse(chainer.should_use_mkldnn('==always'))
            self.assertFalse(chainer.should_use_mkldnn('>=auto'))

testing.run_module(__name__, __file__)