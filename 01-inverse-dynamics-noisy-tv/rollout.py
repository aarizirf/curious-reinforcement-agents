"""Random-action rollouts, in fixed-length episodes."""

import numpy as np
import torch

from corridor import Corridor, N_ACT, N_POS, OBS_DIM

EP_LEN = 100


def collect(condition, n, rng):
    """n transitions as (obs, act, next_obs, tv_bits_at_s)."""
    env = Corridor(slot=condition == "SLOT", rng=rng)
    act = rng.integers(N_ACT, size=n).astype(np.int64)
    obs = np.empty((n, OBS_DIM), dtype=np.float32)
    next_obs = np.empty((n, OBS_DIM), dtype=np.float32)

    for t in range(n):
        if t % EP_LEN == 0:
            env.reset()
        obs[t] = env.obs()
        env.step(act[t])
        next_obs[t] = env.obs()

    tv = obs[:, N_POS:]
    return torch.from_numpy(obs), torch.from_numpy(act), torch.from_numpy(next_obs), tv


def wall_stationary(obs, next_obs):
    """Mask: at a wall and position unchanged, so the action was move-into-wall or press."""
    at_wall = (obs[:, 0] == 1) | (obs[:, N_POS - 1] == 1)
    same_pos = (obs[:, :N_POS] == next_obs[:, :N_POS]).all(1)
    return at_wall & same_pos
