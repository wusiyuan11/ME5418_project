# Code provenance and disclosure for the final project

This file summarizes provenance supported by the available group records. It
does not establish line-by-line authorship or replace an external-code audit.
Each group member should disclose any additional copied or adapted examples,
repositories, licenses and affected files from their development history.

| Component | Provenance or source | Role |
| --- | --- | --- |
| `environment/`, `visualization/` | Group 39 earlier environment milestone | Group project code, extending the earlier simulator |
| `rl/network.py`, `rl/ppo.py`, `train.py` | Group 39 neural-network milestone and subsequent edits | Group PPO/GAE implementation; `INHERITANCE.json` records historical file hashes |
| `pure_rl/`, `run_learning_agent.py`, `configs/` | Group 39 Learning Agent development | 23-input PPO, training wrapper, collection and evaluation |
| `models/foundation_ppo.pt`, `models/mission_ppo.pt` | Group-trained PPO runs, see `models/README.md` | Internal model weights; not external pretrained or expert-demonstration weights |
| PPO algorithm | Schulman et al., https://arxiv.org/abs/1707.06347 | Method reference, not a declaration of copied source |
| PPO reference | OpenAI Spinning Up, https://spinningup.openai.com/en/latest/algorithms/ppo.html | Method documentation and implementation reference; exact copied fragments, if any, require author confirmation |
| GAE | Schulman et al., https://arxiv.org/abs/1506.02438 | Method reference |
| `rl/ppo.py`, `pure_rl/learning.py` | https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py | Public PPO implementation using the same clipped objective; method correspondence, not proof of copying |
| `rl/network.py: _log_probability` | https://github.com/pytorch/pytorch/blob/v2.2.2/torch/distributions/transforms.py#L654-L687 | Numerically stable Tanh Jacobian expression matches `TanhTransform.log_abs_det_jacobian` after variable renaming; the actual development path is unverified |
| PyTorch's cited Tanh derivation | https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/bijectors/tanh.py | Upstream reference cited by PyTorch; no TensorFlow dependency or direct copying is asserted |
| PyTorch | https://pytorch.org/ | Third-party tensor, neural-network, distribution, optimizer and autograd library |
| Gymnasium | https://gymnasium.farama.org/ | Third-party environment API |
| PyBullet | https://github.com/bulletphysics/bullet3 | Third-party physics engine |
| NumPy, Matplotlib, Pillow, PyYAML, pytest, ReportLab | https://numpy.org/, https://matplotlib.org/, https://python-pillow.org/, https://pyyaml.org/, https://pytest.org/, https://www.reportlab.com/ | Third-party numerical, plotting, image, configuration, testing and report libraries |

The available notes say that Stable-Baselines3, gym-pybullet-drones, external
policy weights and expert datasets are not used in the final submitted route.
An earlier local PID-demonstration experiment was a different experiment and
must not be described as the origin of the included PPO-only checkpoint.
Codex assisted implementation, experiments, testing, packaging and writing.
The group is responsible for verifying provenance and complying with all
third-party licenses and the course's originality requirements.

## Notes on source references

`rl/ppo.py` and `pure_rl/learning.py` implement published PPO and GAE methods;
`rl/network.py` contains the Tanh correction formula referenced above. Similar
algorithms or formulas can be implemented independently, so a match alone does
not establish a copying path. The available search and development records are
incomplete; group members should add any other copied or adapted code and its
license. Third-party dependencies installed at runtime are separate software.
