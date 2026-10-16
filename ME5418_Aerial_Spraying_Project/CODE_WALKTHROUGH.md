# Code walkthrough and model explanation

## Initialization

`SprayingEnv.__init__()` defines a 0.1 s step, 20 m square field, 1.2 kg dry mass, payload/spray settings, wind range, illustrative drag scale, acceleration/speed limits, and a 110 s episode limit. The action space is two normalized planar acceleration commands. The observation has nine values: position, velocity, displacement to the active waypoint, route fraction, time fraction, and speed. Wind and mass are diagnostics in `info`, not observations.

## Gymnasium lifecycle

`reset(seed=...)` restores position `[1, 2]`, zero velocity, full liquid, step zero, and target waypoint 1 (the second point because the drone begins at point 0). It samples one wind magnitude and direction for the whole episode and initializes histories. It returns `(observation, info)`.

`step(action)` checks the action shape, clips it to `[-1, 1]`, computes relative-air drag, integrates velocity and position with forward Euler, applies the analytical payload law, tests waypoint/boundary/time conditions, calculates reward, records histories, and returns Gymnasium's five-value tuple.

## Position and wind dynamics

The simplified equations are

```
v_rel = v_drone - v_wind
F_wind = -c_d ||v_rel|| v_rel
a = a_max action + F_wind / m_total
v[k+1] = clip_speed(v[k] + a dt)
p[k+1] = p[k] + v[k+1] dt
```

The wind magnitude is sampled from `Uniform(0, 3)` m/s by default and its angle from `Uniform(0, 2*pi)`. The vector `[V cos(theta), V sin(theta)]` is expressed in the fixed world x-y frame. This is illustrative drag, not calibrated aerodynamics. Mass affects only the acceleration caused by wind force; the normalized control term is already expressed as commanded acceleration.

## Payload model

At simulated time `t`, liquid and total mass are

```
m_liquid(t) = max(m_liquid_initial - spray_rate * t, 0)
m_total(t) = m_drone + m_liquid(t)
```

The analytical update prevents cumulative subtraction error. Liquid cannot become negative, total mass cannot fall below 1.2 kg, and depletion stops at zero.

## Trajectory and controller

`_make_path()` creates five horizontal lanes at y = 2, 6, 10, 14, and 18 m. Their x endpoints alternate between 1 and 19 m, forming ten waypoints and four vertical connectors. The controller maps position error to bounded desired velocity and then maps velocity error to normalized acceleration. It is a PD-style waypoint tracker; it is neither a trained RL policy nor a full attitude PID.

## Termination and reward

The mission terminates successfully within 0.38 m of the last waypoint and terminates as a crash outside the field. It truncates after 1,100 steps if neither occurs. Reward combines distance progress, current target-distance penalty, effort penalty, waypoint bonuses, completion bonus, and crash penalty.

## Visualization and outputs

`render()` creates the final field view. `save_animation()` updates the actual path, drone, target, wind, time, payload, total mass, tracking error, and mission state. `save_static_figures()` creates trajectory, mass, and error figures. `save_results_csv()` stores every state sample. The MP4 branch runs only when FFmpeg is discoverable.

## Important interpretation detail

The plotted "tracking error" is distance to the currently active waypoint. It rises discontinuously when the target advances. It is not cross-track distance to the continuous polyline and should not be used as a final controller-performance metric without adding a segment-based error calculation.
