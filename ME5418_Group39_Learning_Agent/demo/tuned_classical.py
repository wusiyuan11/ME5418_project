"""Stronger classical comparison ONLY; never imported by the pure-PPO trainer.

These previously tuned gains are retained to avoid comparing only with a weak
PID. No trajectories/actions from this controller are used to train pure PPO.
"""
import numpy as np


class TunedClassical:
    def reset(self):
        self.integral_z = 0.

    def __call__(self,observation,env):
        error = observation[12:15]
        velocity_error = env.reference_velocity-observation[3:6]
        self.integral_z = np.clip(self.integral_z+error[2]*env.config.dt,-3.,3.)
        pitch = .6*error[0]+.6*velocity_error[0]
        roll = -(.6*error[1]+.6*velocity_error[1])
        tilt = max(.6,np.cos(observation[6])*np.cos(observation[7]))
        thrust = .8*error[2]+.6*velocity_error[2]+.12*self.integral_z+(1/tilt-1)/env.config.thrust_deviation_fraction
        yaw = -.8*observation[8]-.2*observation[11]
        return np.clip([thrust,roll,pitch,yaw],-.98,.98).astype(np.float32)
