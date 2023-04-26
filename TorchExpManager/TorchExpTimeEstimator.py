import datetime
from typing import Optional

from .utils import EMATimeEstimator, format_time


class ExpTimeEstimator:
    def __init__(self, num_train, num_valid):
        self.train_time_estimator = EMATimeEstimator(num_train)
        self.valid_time_estimator = EMATimeEstimator(num_valid)

    def update_train(self, x):
        self.train_time_estimator(x)

    def update_valid(self, x):
        self.valid_time_estimator(x)

    @property
    def elapsed_time(self):
        return self.train_time_estimator.elapsed_time + self.valid_time_estimator.elapsed_time

    @property
    def x(self):
        return self.train_time_estimator.x + self.valid_time_estimator.x

    @property
    def remaining_time(self):
        return self.train_time_estimator.remaining_time + self.valid_time_estimator.remaining_time

    @property
    def total_time(self):
        return self.train_time_estimator.total_time + self.valid_time_estimator.total_time

    def print_status(self, remaining_train: Optional[int] = None, remaining_valid: Optional[int] = None):
        expected_end_time = datetime.datetime.now()
        if remaining_train is not None and remaining_valid is not None:
            expected_end_time += datetime.timedelta(seconds=(self.train_time_estimator.average_speed * remaining_train +
                                                             self.valid_time_estimator.average_speed * remaining_valid))
        else:
            expected_end_time += datetime.timedelta(seconds=self.remaining_time)
        print(
            f"Time: {format_time(self.x)} -> {format_time(self.elapsed_time)}/{format_time(self.total_time)}"
            f"{'+?' if self.valid_time_estimator.elapsed_calls == 0 else ''}"
            f" | Expected end Time: "
            f"{expected_end_time.strftime('%Y-%m-%d %H:%M:%S')}"
            f"{'+?' if self.valid_time_estimator.elapsed_calls == 0 else ''}")
