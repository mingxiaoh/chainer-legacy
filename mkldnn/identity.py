from chainer import function

# Mainly for rank/fanout increase
# Forward only
class Identity(function.Function):
    def forward(self, x):
        y = x
        return y,
