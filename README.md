# Curious reinforcement agents

Agents that learn in environments with sparse rewards have a hard problem: most of
what they do produces no feedback at all. One answer is to let the agent generate
its own reward — to make it curious, and let curiosity carry it through the long
stretches where the environment says nothing.

This repository is the working record of that investigation. Each numbered folder
is one self-contained step: the code, the results, and a written post explaining
what we were asking and what came back. They run on CPU in minutes.

| step | question | status |
|---|---|---|
| [`01-inverse-dynamics-noisy-tv`](01-inverse-dynamics-noisy-tv) | Does an inverse-dynamics encoder really ignore environment noise? | done |

Written up at [aarizirfan.com/projects/curious-rl-agents](https://aarizirfan.com/projects/curious-rl-agents).
