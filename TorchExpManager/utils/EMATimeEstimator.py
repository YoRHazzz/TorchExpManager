class EMA(object):
    """
    Exponential moving average: smoothing to give progressively lower
    weights to older values.
    Parameters
    ----------
    smoothing  : float, optional
        Smoothing factor in range [0, 1], [default: 0.3].
        Increase to give more weight to recent values.
        Ranges from 0 (yields old value) to 1 (yields new value).
    """

    def __init__(self, smoothing=0.3):
        self.alpha = smoothing
        self.beta = 1 - self.alpha
        self.last = 0
        self.calls = 0
        self.average = 0

    def __call__(self, x=None):
        """
        Parameters
        ----------
        x  : float
            New value to include in EMA.
        """
        if x is not None:
            self.last = self.alpha * x + self.beta * self.last
            self.calls += 1
        self.average = self.last / (1 - self.beta ** self.calls) if self.calls else self.last
        return self


class EMATimeEstimator:
    def __init__(self, total_calls, smoothing=0.3):
        self.ema_time = EMA(smoothing)
        self.elapsed_time = 0
        self.remaining_time = 0
        self.elapsed_calls = 0
        self.total_calls = total_calls
        self.x = 0

    @property
    def remaining_calls(self):
        return self.total_calls - self.elapsed_calls

    def __call__(self, x=None):
        if x is not None and self.remaining_calls > 0:
            self.x = x
            self.ema_time(x)
            self.elapsed_time += x
            self.elapsed_calls += 1
            self.remaining_time = self.ema_time.average * self.remaining_calls

    @property
    def total_time(self):
        return self.elapsed_time + self.remaining_time

    @property
    def average_speed(self):
        return self.ema_time.average
