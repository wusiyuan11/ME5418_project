# ME5418 Aerial Spraying Gymnasium Prototype

This project is a reproducible **2-D point-mass** research scaffold for studying trajectory tracking under horizontal wind and decreasing spray payload. It is not a full 3-D quadrotor, does not model attitude or rotors, and does not yet train PPO.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_environment.py
pytest -q
python spraying_env.py --seed 7 --output outputs
```

The run creates an animated GIF, trajectory comparison, payload-mass plot, active-waypoint-error plot, and numerical CSV. It also creates an MP4 when FFmpeg is installed.

## Model scope

The controller issues normalized planar acceleration. An illustrative quadratic relative-air drag force is divided by current total mass, so payload changes alter wind-induced acceleration. Wind is constant within an episode and sampled in world coordinates. See [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md) and [USER_GUIDE.txt](USER_GUIDE.txt) for details.

## Research roadmap

- G1: no wind, fixed mass.
- G2: no wind, decreasing mass.
- G3: random wind, fixed mass.
- G4: random wind and decreasing mass.
- G5: unseen wind and spraying conditions.

PID and PPO comparisons belong after this base environment is validated. Gain scheduling and recurrent PPO are later extensions, not current results.
