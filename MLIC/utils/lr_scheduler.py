import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineWithFloor(_LRScheduler):
    """
    Cosine decay from base_lr (the optimizer's initial lr) down to eta_min
    over T epochs, then holds constant at eta_min.
    """
    def __init__(self, optimizer, T, eta_min=0.0, last_epoch=-1):
        self.T = T
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.T:
            return [self.eta_min for _ in self.base_lrs]

        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * self.last_epoch / self.T)) / 2
            for base_lr in self.base_lrs
        ]


class SmoothCosineDecay(_LRScheduler):
    """
    Continuous cosine oscillation (no restart discontinuity):
      - warmup_t epochs: linear ramp from warmup_lr_init -> base_lr
      - after that: lr oscillates as a full cosine wave (period = t_initial),
        with peak envelope shrinking smoothly by `cycle_decay` every t_initial
        epochs (continuous exponential decay, so no jump at cycle boundaries),
        floor at lr_min.
    """
    def __init__(self, optimizer, t_initial, lr_min=0.0,
                 warmup_t=0, warmup_lr_init=0.0,
                 cycle_decay=1.0, cycle_limit=None, last_epoch=-1):
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit  # max number of periods before freezing envelope
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch

        # --- warmup phase ---
        if self.warmup_t > 0 and e < self.warmup_t:
            alpha = e / self.warmup_t
            return [
                self.warmup_lr_init + alpha * (base_lr - self.warmup_lr_init)
                for base_lr in self.base_lrs
            ]

        t = e - self.warmup_t  # time since warmup ended

        # continuous time in units of "number of periods elapsed"
        n = t / self.t_initial
        if self.cycle_limit is not None:
            n = min(n, self.cycle_limit)

        lrs = []
        for base_lr in self.base_lrs:
            envelope = base_lr * (self.cycle_decay ** n)  # smooth exponential decay of peak
            cos_wave = (1 + math.cos(2 * math.pi * t / self.t_initial)) / 2  # full period: 1 -> 0 -> 1
            lr = self.lr_min + (envelope - self.lr_min) * cos_wave
            lrs.append(lr)
        return lrs