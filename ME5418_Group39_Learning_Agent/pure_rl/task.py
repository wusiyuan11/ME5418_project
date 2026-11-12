"""Observable mission state and optional training-only reinforcement curriculum.

The base PyBullet environment, 2 m/s route, 0.5 m waypoint tolerance and success
test are unchanged. Evaluation always uses its original reset distribution.
Reference velocity / next waypoint are public task commands, not expert actions.
True wind and tank mass never enter the policy observation.
"""
from dataclasses import dataclass
import gymnasium as gym
import numpy as np


def mission_observation(observation, env, features='mission'):
    """Expose mission history to the policy, not just the moving reference.

    Feature layout (zero-based): [0:15] inherited state/tracking observation;
    [15:18] reference velocity, or velocity error in the submitted 'goal' mode;
    [18:21] next outstanding waypoint minus vehicle position; [21] ordered
    arrival count / waypoint count; [22] remaining mission-time fraction.

    The previous submission already counted arrivals internally. The milestone
    change is making that count and its implied next goal observable to BOTH
    actor and critic. No wind, tank mass, expert action or critic output is added.
    """
    # If c waypoints have been visited in order, waypoint[c] is still owed.
    # Use arrival bookkeeping, NOT current_waypoint/progress from the clock.
    # After all arrivals, retain the final goal rather than indexing past it.
    index = min(env.waypoints_achieved, len(env.trajectory.waypoints)-1)
    next_error = env.trajectory.waypoints[index] - observation[:3]
    command = env.reference_velocity if features=='mission' else env.reference_velocity-observation[3:6]
    # During normal full-start episodes, c increases only after the original
    # sampled 3-D arrival test. Curriculum prefixes are declared at reset and
    # logged separately below; they must not be presented as earned arrivals.
    return np.concatenate((observation, command, next_error,
        [env.waypoints_achieved / len(env.trajectory.waypoints),
         max(0., 1-env.time/env.config.max_episode_time)])).astype(np.float32)


@dataclass
class TaskOptions:
    """Training-only choices; selected values live in configs/mission_ppo.json.

    These historical defaults differ from the selected submission preset.
    Evaluation never uses this wrapper's shaping or curriculum distribution.
    """
    reward: str = 'aligned'
    curriculum: bool = False
    progress_weight: float = .5
    waypoint_bonus: float = 1.
    success_bonus: float = 8.
    timeout_penalty: float = 4.
    original_reward_scale: float = .1
    features: str = 'mission'
    action_repeat: int = 1
    early_miss_abort: bool = False
    goal_distance_weight: float = 0.

    def validate(self):
        if self.reward not in ('original', 'aligned'):
            raise ValueError('Unknown reinforcement reward')
        if self.features not in ('mission','goal') or self.action_repeat<1:
            raise ValueError('Invalid public features/action repeat')
        values = (self.progress_weight, self.waypoint_bonus, self.success_bonus,
                  self.timeout_penalty, self.original_reward_scale,self.goal_distance_weight)
        if not all(np.isfinite(value) and value >= 0 for value in values):
            raise ValueError('Reward weights must be finite and nonnegative')


class LearningTask(gym.Wrapper):
    """Training wrapper only. No controller is used for reset or reward."""
    def __init__(self, env, options):
        super().__init__(env)
        options.validate()
        self.options = options
        self.fraction = 0.
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(23,), dtype=np.float32)

    def set_fraction(self, fraction):
        self.fraction = float(np.clip(fraction, 0, 1))

    def potential(self):
        # Scalar route-aware progress towards the outstanding goal. This is
        # heuristic shaping, not a claim of policy-invariant PBRS: the code
        # adds Phi(next)-Phi(current), without a gamma factor or a proof.
        env = self.unwrapped
        index = min(env.waypoints_achieved, len(env.trajectory.waypoints)-1)
        return float(env.trajectory.cumulative_lengths[index] -
                     np.linalg.norm(env.dynamics.position-env.trajectory.waypoints[index]))

    def reset(self, **kwargs):
        env = self.unwrapped
        # Curriculum changes only at reset. Last 25% of training is the exact
        # original start-at-rest / full-wind distribution, not partial flights.
        probability = .8 * max(0., 1-self.fraction/.75) if self.options.curriculum else 0.
        wind_limit = .5 + 2.5*min(1., self.fraction/.5) if self.options.curriculum else 3.
        env.wind.max_speed = wind_limit
        observation, info = self.env.reset(**kwargs)
        start_time = 0.
        if env.np_random.random() < probability:
            times = env.trajectory.cumulative_lengths[1:-1] / env.config.reference_speed
            # Public reference states, not states/actions generated by PID.
            start_time = max(0., float(env.np_random.choice(times) - env.np_random.uniform(.4, 1.6)))
            start_time = round(start_time / env.config.dt) * env.config.dt
            env.time, env.step_count = start_time, round(start_time/env.config.dt)
            env.reference_position, env.reference_velocity, env.current_waypoint, env.progress = \
                env.trajectory.reference(start_time)
            env.payload.update(start_time)
            position = env.reference_position + env.np_random.uniform(-.08, .08, 3)
            env.dynamics.reset(position, env.payload.total_mass)
            env.dynamics.set_state(position, env.reference_velocity * env.np_random.uniform(.7, 1.1))
            distance = start_time * env.config.reference_speed
            # This subtask assumes a route prefix. It is not a physical visit
            # demonstration and earns no reset reward. Save the initial count
            # so the collector can report only newly earned waypoints.
            env.waypoints_achieved = int(np.searchsorted(env.trajectory.cumulative_lengths, distance,
                                                         side='right'))
            env._check_waypoint_arrival()
            observation = env._observation()
            info = env._info('running', {})
        self.start_reference_time = start_time
        self.start_waypoints = env.waypoints_achieved
        self.active_wind_limit = wind_limit
        self.previous_potential = self.potential()
        info.update({'reset_reference_time': start_time, 'curriculum_wind_limit': wind_limit})
        return mission_observation(observation, env,self.options.features), info

    def _physics_step(self, action):
        env = self.unwrapped
        previous_count = env.waypoints_achieved
        # The inherited environment, not the wrapper, decides real arrivals,
        # crashes, boundaries and success. Its reference still advances by time.
        observation, original, terminal, truncated, info = self.env.step(action)
        reward = self.options.original_reward_scale * original
        if self.options.reward == 'aligned':
            current = self.potential()
            reward += self.options.progress_weight * (current-self.previous_potential)
            # Reward a physical arrival event, not reference-time advancement
            # or the absolute count (which would pay repeatedly for old visits).
            reward += self.options.waypoint_bonus * (env.waypoints_achieved-previous_count)
            reward += self.options.success_bonus * info['is_success']
            reward -= self.options.timeout_penalty * truncated
            reward -= .01 * float(observation[8]**2) * env.config.dt
            index = min(env.waypoints_achieved,len(env.trajectory.waypoints)-1)
            reward -= self.options.goal_distance_weight * min(12.,float(np.linalg.norm(
                env.trajectory.waypoints[index]-observation[:3]))) * env.config.dt
            self.previous_potential = current
        cutoff = False
        # Early training concentrates attempts around missed turns; withdraw
        # this artificial terminal for the final quarter. Never use it in test.
        if self.options.early_miss_abort and self.fraction<.75 and not (terminal or truncated):
            index = env.waypoints_achieved
            if index<len(env.trajectory.waypoints):
                target_time = env.trajectory.cumulative_lengths[index]/env.config.reference_speed
                cutoff = env.time>target_time+1.5
                if cutoff:
                    reward -= self.options.timeout_penalty
                    info['termination_reason'] = 'training_waypoint_lag'
        info['original_reward'] = original
        info['reset_reference_time'] = self.start_reference_time
        # The 45 s deadline defines task failure, rather than an arbitrary
        # interruption of an infinite-horizon task. Remaining time is observed.
        return mission_observation(observation, env,self.options.features), float(reward), terminal or truncated or cutoff, False, info

    def step(self,action):
        # A 10 Hz outer-loop action holds for two 20 Hz physics steps in the
        # submitted preset. Arrival/safety checks still run at EVERY microstep,
        # and a terminal stops the hold immediately. Log the actual step count.
        total_reward,total_original,error_square = 0.,0.,0.
        for count in range(self.options.action_repeat):
            observation,reward,terminal,truncated,info = self._physics_step(action)
            total_reward += reward
            total_original += info['original_reward']
            error_square += info['tracking_error_norm']**2
            if terminal or truncated:
                break
        info['original_reward'] = total_original
        info['physics_steps'] = count+1
        info['tracking_error_square_sum'] = error_square
        return observation,total_reward,terminal,truncated,info
