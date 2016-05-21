from chainer.function_hooks import debug_print
from chainer.function_hooks import timer

PrintHook = debug_print.PrintHook
AccumulateTimerHook = timer.AccumulateTimerHook
TimerHook = timer.TimerHook
