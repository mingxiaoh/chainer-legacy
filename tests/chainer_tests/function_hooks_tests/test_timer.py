import time
import unittest

import numpy

import chainer
from chainer import cuda
from chainer import function_hooks
from chainer import functions
from chainer.functions.connection import linear
from chainer import links
from chainer import testing
from chainer.testing import attr


def check_history(self, t, function_type, return_type):
    self.assertIsInstance(t[0], function_type)
    self.assertIsInstance(t[1], return_type)


@testing.parameterize(
    {'hook': function_hooks.TimerHook, 'name': 'TimerHook'},
    {'hook': function_hooks.AccumulateTimerHook, 'name': 'AccumulateTimerHook'}
)
class TestTimerHookToLink(unittest.TestCase):

    def setUp(self):
        self.h = self.hook()
        self.l = links.Linear(5, 5)
        self.x = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)

    def test_name(self):
        self.assertEqual(self.h.name, self.name)

    def check_forward(self, x):
        with self.h:
            self.l(chainer.Variable(x))
        self.assertEqual(1, len(self.h.call_history))
        check_history(self, self.h.call_history[0],
                      linear.LinearFunction, float)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.l.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x, gy):
        x = chainer.Variable(x)
        y = self.l(x)
        y.grad = gy
        with self.h:
            y.backward()
        self.assertEqual(1, len(self.h.call_history))
        check_history(self, self.h.call_history[0],
                      linear.LinearFunction, float)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.l.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


@testing.parameterize(
    {'hook': function_hooks.TimerHook, 'name': 'TimerHook'},
    {'hook': function_hooks.AccumulateTimerHook, 'name': 'AccumulateTimerHook'}
)
class TestTimerHookToFunction(unittest.TestCase):

    def setUp(self):
        self.h = function_hooks.TimerHook()
        self.f = functions.Exp()
        self.f.add_hook(self.h)
        self.x = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)

    def check_forward(self, x):
        self.f(chainer.Variable(x))
        self.assertEqual(1, len(self.h.call_history))
        check_history(self, self.h.call_history[0], functions.Exp, float)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_fowward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x, gy):
        x = chainer.Variable(x)
        y = self.f(x)
        y.grad = gy
        y.backward()
        self.assertEqual(2, len(self.h.call_history))
        check_history(self, self.h.call_history[1], functions.Exp, float)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class TestAccumulateTimerHook(unittest.TestCase):

    def setUp(self):
        self.h = function_hooks.AccumulateTimerHook()

    def test_context(self):
        self.assertEqual(self.h.current_context, self.h.default_context)
        with self.h('a') as h:
            self.assertEqual(h.current_context, 'a')
        self.assertEqual(self.h.current_context, self.h.default_context)
        self.h('b')
        self.assertEqual(self.h.current_context, 'b')
        self.h()
        self.assertEqual(self.h.current_context, self.h.default_context)

    def test_accumulate(self):
        with self.h('a'):
            time.sleep(0.1)
        numpy.testing.assert_allclose(self.h.total_time('a'), 0.1, rtol=0.1)
        with self.h('a'):
            time.sleep(0.1)
        numpy.testing.assert_allclose(self.h.total_time('a'), 0.2, rtol=0.1)
        with self.h('b'):
            time.sleep(0.1)
        numpy.testing.assert_allclose(self.h.total_time('b'), 0.1, rtol=0.1)


testing.run_module(__name__, __file__)
