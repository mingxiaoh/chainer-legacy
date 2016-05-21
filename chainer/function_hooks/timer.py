import time
import collections

import numpy

from chainer import cuda
from chainer import function


class Timer(object):

    def __init__(self, xp):
        self.xp = xp
        self.running = False
        self.total_time = 0.0
        self.last_increment = None

    def start(self):
        if self.running:
            return
        if self.xp == numpy:
            self._start = time.time()
        else:
            self._start = cuda.Event()
            self._start.record()
        self.running = True

    def stop(self):
        if not self.running:
            return 0.0

        if self.xp == numpy:
            self._stop = time.time()
            elapsed_time = self._stop - self._start
        else:
            self._stop = cuda.Event()
            self._stop.record()
            self._stop.synchronize()
            # Note that `get_elapsed_time` returns result in milliseconds
            elapsed_time = cuda.cupy.cuda.get_elapsed_time(
                self._start, self._stop) / 1000
        self.running = False
        self.total_time += elapsed_time
        self.last_increment = elapsed_time
        return elapsed_time


class TimerHook(function.FunctionHook):
    """Function hook for measuring elapsed time of functions.

    Attributes:
        call_history: List of measurement results. It consists of pairs of
            the function that calls this hook and the elapsed time
            the function consumes.
    """

    name = 'TimerHook'

    def __init__(self):
        self.call_history = []

    def _preprocess(self):
        self.timer = Timer(self.xp)
        self.timer.start()

    def forward_preprocess(self, function, in_data):
        self.xp = cuda.get_array_module(*in_data)
        self._preprocess()

    def backward_preprocess(self, function, in_data, out_grad):
        self.xp = cuda.get_array_module(*(in_data + out_grad))
        self._preprocess()

    def _postprocess(self, function):
        elapsed_time = self.timer.stop()
        self.call_history.append((function, elapsed_time))

    def forward_postprocess(self, function, in_data):
        xp = cuda.get_array_module(*in_data)
        assert xp == self.xp
        self._postprocess(function)

    def backward_postprocess(self, function, in_data, out_grad):
        xp = cuda.get_array_module(*(in_data + out_grad))
        assert xp == self.xp
        self._postprocess(function)

    def total_time(self):
        """Returns total elapsed time in seconds."""
        return sum(t for (_, t) in self.call_history)


class AccumulateTimerHook(function.FunctionHook):

    name = 'AccumulateTimerHook'
    default_context = 'default'

    def __init__(self, xp=numpy):
        self.xp = xp
        self.timers = collections.defaultdict(lambda :Timer(self.xp))
        self._call_history = collections.defaultdict(list)
        self.current_context = self.default_context

    @property
    def call_history(self, context=None):
        return self._call_history[self.current_context]

    def __call__(self, context=None):
        if context:
            self.current_context = context
        else:
            self.current_context = self.default_context
        return self

    def __enter__(self, *args, **kwargs):
        self.timers[self.current_context].start()
        return super(AccumulateTimerHook, self).__enter__(*args, **kwargs)

    def __exit__(self, *args, **kwargs):
        self.timers[self.current_context].stop()
        self.current_context = self.default_context
        return super(AccumulateTimerHook, self).__exit__(*args, **kwargs)

    def _preprocess(self, xp):
        self.function_call_timer = Timer(xp)

    def forward_preprocess(self, function, in_data):
        xp = cuda.get_array_module(*in_data)
        self._preprocess(xp)

    def backward_preprocess(self, function, in_data, out_grad):
        xp = cuda.get_array_module(*(in_data + out_grad))
        self._preprocess(xp)

    def _postprocess(self, function):
        elapsed_time = self.function_call_timer.stop()
        self._call_history[self.current_context].append((function, elapsed_time))

    def forward_postprocess(self, function, in_data):
        xp = cuda.get_array_module(*in_data)
        assert xp == self.function_call_timer.xp
        self._postprocess(function)

    def backward_postprocess(self, function, in_data, out_grad):
        xp = cuda.get_array_module(*(in_data + out_grad))
        assert xp == self.function_call_timer.xp
        self._postprocess(function)

    def total_time(self, context=None):
        """Returns total elapsed time in seconds of given context."""

        if not context:
            context = self.current_context

        if context in self.timers:
            return self.timers[context].total_time
        else:
            raise ValueError('No such context:{}'.format(context))

    def function_call_time(self, context=None):
        """Returns total elapsed time of function calls in given context in seconds."""

        if not context:
            context = self.current_context

        if context in self.timers:
            return sum(t for (_, t) in self._call_history[context])
        else:
            raise ValueError('No such context:{}'.format(context))

    def last_increment(self, context=None):

        if not context:
            context = self.current_context

        if context in self.timers:
            return self.timers[context].last_increment
        else:
            raise ValueError('No such context:{}'.format(context))
