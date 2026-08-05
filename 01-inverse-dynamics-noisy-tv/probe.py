"""Read information back out of a frozen encoder. Chance is 0.5 throughout."""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from corridor import PRESS

N_TRAIN = 10_000
N_TEST = 5_000


def content(phi, obs, tv, n_bits, nonlinear=False):
    """Mean per-bit accuracy recovering the TV bits from phi(s)."""
    with torch.no_grad():
        z = phi(obs).numpy()
    accs = []
    for b in range(n_bits):
        clf = (
            MLPClassifier((64, 64), max_iter=300)
            if nonlinear
            else LogisticRegression(max_iter=500)
        )
        clf.fit(z[:N_TRAIN], tv[:N_TRAIN, b])
        accs.append(clf.score(z[N_TRAIN:], tv[N_TRAIN:, b]))
    return float(np.mean(accs))


def change(phi, obs, next_obs, act):
    """Accuracy detecting a TV resample from the pair (phi(s), phi(s')). SLOT only."""
    with torch.no_grad():
        z = torch.cat([phi(obs), phi(next_obs)], dim=1).numpy()
    resampled = (act.numpy() == PRESS)
    clf = MLPClassifier((64,), max_iter=300).fit(z[:N_TRAIN], resampled[:N_TRAIN])
    return clf.score(z[N_TRAIN:], resampled[N_TRAIN:])
