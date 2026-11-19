# Limited public-source traceability audit

## Identified public references

The review identified the following correspondences to public formulas or
algorithm implementations:

| Project file | What was matched | Public reference | Evidence level |
| --- | --- | --- | --- |
| `rl/network.py`, `_log_probability` | The numerically stable Tanh log-Jacobian expression has the same structure as PyTorch's transform after variable renaming. | https://github.com/pytorch/pytorch/blob/v2.2.2/torch/distributions/transforms.py#L654-L687 | Specific mathematical-expression correspondence; copying path unknown |
| `rl/ppo.py` | GAE and the clipped PPO objective implement published algorithms; their equations can be compared with Spinning Up. | https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py | Algorithm correspondence, not evidence that the file was copied |
| `pure_rl/learning.py` | The clipped PPO surrogate implements the same published method. | https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py | Algorithm correspondence, not evidence that the file was copied |

PPO and GAE primary method sources are https://arxiv.org/abs/1707.06347 and
https://arxiv.org/abs/1506.02438. The PyTorch source cites the Tanh bijector in
https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/bijectors/tanh.py.
The group also uses PyTorch, Gymnasium, PyBullet and other third-party packages
through imports; these packages are not vendored project files and their source
lines are excluded from the denominator. See `CODE_PROVENANCE.md` for the
broader dependency list and project history.

The review used selective searches and available records; it was not an
exhaustive clone-detection or authorship investigation. A formula match is not
the same as an entire copied file, and standard equations are not sufficient
evidence of copying. Group members must disclose actual copied or adapted code,
AI assistance and applicable licenses based on their development history, and
use the course's required verification method.
