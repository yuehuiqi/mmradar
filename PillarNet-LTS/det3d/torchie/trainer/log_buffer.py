from collections import OrderedDict

import numpy as np


class LogBuffer(object):
    def __init__(self):
        self.val_history = OrderedDict()
        self.n_history = OrderedDict()
        self.output = OrderedDict()
        self.ready = False

    def clear(self):
        self.val_history.clear()
        self.n_history.clear()
        self.clear_output()

    def clear_output(self):
        self.output.clear()
        self.ready = False

    def update(self, vars, count=1):
        assert isinstance(vars, dict)
        for key, var in vars.items():
            if key not in self.val_history:
                self.val_history[key] = []
                self.n_history[key] = []
            self.val_history[key].append(var)
            self.n_history[key].append(count)

    def average(self, n=0):
        """Average latest n values or all values"""
        assert n >= 0
        for key in self.val_history:
            raw_values = self.val_history[key][-n:]
            try:
                values = np.asarray(raw_values, dtype=np.float64)
            except ValueError:
                last_value = raw_values[-1]
                self.output[key] = last_value.tolist() if hasattr(last_value, "tolist") else last_value
                continue
            nums = np.asarray(self.n_history[key][-n:], dtype=np.float64)
            if values.shape == nums.shape:
                avg = np.sum(values * nums) / np.sum(nums)
            elif values.shape[0] == nums.shape[0]:
                weight_shape = (nums.shape[0],) + (1,) * (values.ndim - 1)
                avg = np.sum(values * nums.reshape(weight_shape), axis=0) / np.sum(nums)
                avg = avg.tolist()
            else:
                avg = np.mean(values, axis=0).tolist()
            self.output[key] = avg
        self.ready = True
