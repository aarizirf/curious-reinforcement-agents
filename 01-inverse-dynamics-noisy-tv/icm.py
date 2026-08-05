"""ICM's encoder and inverse dynamics model, trained on the inverse loss alone."""

import torch
import torch.nn as nn

from corridor import N_ACT, OBS_DIM
from rollout import wall_stationary

LR = 3e-3
EPOCHS = 15
BATCH = 256


def make_encoder(emb_dim):
    return nn.Sequential(nn.Linear(OBS_DIM, 64), nn.ELU(), nn.Linear(64, emb_dim))


class Inverse(nn.Module):
    """Predicts the action from a pair of embeddings."""

    def __init__(self, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * emb_dim, 64), nn.ELU(), nn.Linear(64, N_ACT)
        )

    def forward(self, z, z_next):
        return self.net(torch.cat([z, z_next], dim=1))


def train(phi, inv, obs, act, next_obs, log=None):
    opt = torch.optim.Adam([*phi.parameters(), *inv.parameters()], lr=LR)
    ce = nn.CrossEntropyLoss()
    n = len(obs)

    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        total_loss, correct = 0.0, 0
        for i in range(0, n, BATCH):
            idx = perm[i : i + BATCH]
            logits = inv(phi(obs[idx]), phi(next_obs[idx]))
            loss = ce(logits, act[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == act[idx]).sum().item()
        if log:
            log(epoch, total_loss / n, correct / n)


def evaluate(phi, inv, obs, act, next_obs):
    """(accuracy on all transitions, accuracy on wall-stationary ones)."""
    with torch.no_grad():
        correct = inv(phi(obs), phi(next_obs)).argmax(1) == act
    wall = wall_stationary(obs, next_obs)
    return correct.float().mean().item(), correct[wall].float().mean().item()
