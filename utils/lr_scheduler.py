import warnings

import numpy as np
from torch.optim.lr_scheduler import ExponentialLR


class ExponentialLRxyz(ExponentialLR):
    def __init__(
            self,
            optimizer,
            start_step,
            stop_step,
            start_value,
            stop_value,
            group_name,
            verbose=False
    ):
        self._start_step = start_step
        self._stop_step = stop_step
        self._start_value = start_value
        self._stop_value = stop_value
        self._group_name = group_name
        super().__init__(optimizer, 0, -1, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        step = self._step_count
        if step > self._start_step:
            if step < self._stop_step:
                t = step - self._start_step
                t = np.clip(t / (self._stop_step - self._start_step), 0, 1)
                log_lerp = np.exp(np.log(self._start_value) * (1 - t) + np.log(self._stop_value) * t)
            else:
                log_lerp = self._stop_value
        else:
            log_lerp = 0

        target_params = next(item for item in self.optimizer.param_groups if item["name"] == self._group_name)
        target_params['lr'] = log_lerp

        #for group in self.optimizer.param_groups:
        #    if group["name"] == self._group_name:
        #        print(group['lr'])

        return [group['lr'] for group in self.optimizer.param_groups]
