"""8-cell corridor with a K-bit noisy TV."""

import numpy as np

N_POS = 8
K_BITS = 32
N_ACT = 3
OBS_DIM = N_POS + K_BITS

LEFT, RIGHT, PRESS = 0, 1, 2


class Corridor:
    def __init__(self, slot, rng):
        self.slot = slot  # if False, press is a no-op
        self.rng = rng
        self.reset()

    def reset(self):
        self.pos = int(self.rng.integers(N_POS))
        self.bits = self.rng.integers(2, size=K_BITS).astype(np.float32)

    def step(self, a):
        if a == LEFT:
            self.pos = max(0, self.pos - 1)
        elif a == RIGHT:
            self.pos = min(N_POS - 1, self.pos + 1)
        else:
            if self.slot:
                self.bits = self.rng.integers(2, size=K_BITS).astype(np.float32)

    def obs(self):
        """one_hot(pos) ++ bits"""
        o = np.zeros(OBS_DIM, dtype=np.float32)
        o[self.pos] = 1.0
        o[N_POS:] = self.bits
        return o
