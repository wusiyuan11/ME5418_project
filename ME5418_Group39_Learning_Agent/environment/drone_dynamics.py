"""PyBullet rigid-body dynamics for the aerial spraying quadrotor.

The environment sends high-level thrust and attitude commands. This module
turns them into forces and torques on a single rigid body, then lets PyBullet
integrate gravity, translation, rotation and ground contact.
"""

from dataclasses import dataclass
import numpy as np
import pybullet as p


@dataclass
class DroneDynamics:
    """Manage one PyBullet client and the quadrotor rigid body inside it."""

    dt: float = 0.05
    gui: bool = False
    gravity: float = 9.81
    attitude_kp: float = 0.32
    attitude_kd: float = 0.10
    yaw_rate_gain: float = 0.08

    def __post_init__(self) -> None:
        connection_mode = p.GUI if self.gui else p.DIRECT
        self.client = p.connect(connection_mode)
        if self.client < 0:
            raise RuntimeError("PyBullet could not create a physics client")
        self.body_id = -1
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.attitude = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.current_mass = 1.0
        self._configure_world()

    def _configure_world(self) -> None:
        """Create a clean SI-unit world with gravity and a collision floor."""
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0.0, 0.0, -self.gravity, physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)
        p.setPhysicsEngineParameter(fixedTimeStep=self.dt, numSolverIterations=50,
                                    deterministicOverlappingPairs=1,
                                    physicsClientId=self.client)
        floor_shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.client)
        p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=floor_shape,
                          basePosition=[0.0, 0.0, 0.0], physicsClientId=self.client)

    @staticmethod
    def _box_inertia(mass: float) -> np.ndarray:
        """Approximate the airframe and centred tank as a rectangular body."""
        length, width, height = 0.35, 0.35, 0.12
        return mass * np.array([(width**2 + height**2) / 12.0,
                                (length**2 + height**2) / 12.0,
                                (length**2 + width**2) / 12.0])

    def reset(self, position: np.ndarray, mass: float) -> None:
        """Recreate the vehicle at rest for a new episode."""
        self._configure_world()
        half_extents = [0.175, 0.175, 0.06]
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents,
                                           physicsClientId=self.client)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                     rgbaColor=[0.10, 0.55, 0.72, 1.0],
                                     physicsClientId=self.client)
        self.body_id = p.createMultiBody(baseMass=float(mass),
                                         baseCollisionShapeIndex=collision,
                                         baseVisualShapeIndex=visual,
                                         basePosition=np.asarray(position, dtype=float),
                                         physicsClientId=self.client)
        p.changeDynamics(self.body_id, -1, linearDamping=0.05, angularDamping=0.08,
                         lateralFriction=0.8, restitution=0.0,
                         physicsClientId=self.client)
        self.update_mass(mass)
        self._sync_state()

    def update_mass(self, mass: float) -> None:
        """Send the payload-dependent mass and inertia to PyBullet."""
        self.current_mass = float(mass)
        p.changeDynamics(self.body_id, -1, mass=self.current_mass,
                         localInertiaDiagonal=self._box_inertia(self.current_mass).tolist(),
                         physicsClientId=self.client)

    def set_state(self, position: np.ndarray, velocity: np.ndarray | None = None,
                  attitude: np.ndarray | None = None,
                  angular_velocity: np.ndarray | None = None) -> None:
        """Set an exact state, mainly for safety and termination tests."""
        attitude = np.zeros(3) if attitude is None else np.asarray(attitude, dtype=float)
        quaternion = p.getQuaternionFromEuler(attitude.tolist())
        p.resetBasePositionAndOrientation(self.body_id, np.asarray(position, dtype=float),
                                          quaternion, physicsClientId=self.client)
        p.resetBaseVelocity(self.body_id,
                            np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float),
                            np.zeros(3) if angular_velocity is None else np.asarray(angular_velocity, dtype=float),
                            physicsClientId=self.client)
        self._sync_state()

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def step(self, thrust: float, commanded_attitude: np.ndarray, yaw_rate: float,
             mass: float, wind_force: np.ndarray, dt: float) -> None:
        """Apply forces and control torques, then advance PyBullet one step."""
        if abs(dt - self.dt) > 1e-12:
            self.dt = float(dt)
            p.setTimeStep(self.dt, physicsClientId=self.client)
        self.update_mass(mass)
        position, quaternion = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client)
        roll, pitch, _ = p.getEulerFromQuaternion(quaternion)
        _, world_omega = p.getBaseVelocity(self.body_id, physicsClientId=self.client)
        rotation = np.asarray(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
        body_omega = rotation.T @ np.asarray(world_omega)
        torque_body = np.array([
            self.attitude_kp * self._wrap_angle(float(commanded_attitude[0]) - roll)
            - self.attitude_kd * body_omega[0],
            self.attitude_kp * self._wrap_angle(float(commanded_attitude[1]) - pitch)
            - self.attitude_kd * body_omega[1],
            self.yaw_rate_gain * (yaw_rate - body_omega[2]),
        ])
        # PyBullet rotates LINK_FRAME thrust and torque with the vehicle. Wind
        # is already a world-frame force, so it is applied separately.
        p.applyExternalForce(self.body_id, -1, [0.0, 0.0, float(thrust)], [0.0, 0.0, 0.0],
                             p.LINK_FRAME, physicsClientId=self.client)
        p.applyExternalForce(self.body_id, -1, np.asarray(wind_force, dtype=float).tolist(),
                             position, p.WORLD_FRAME, physicsClientId=self.client)
        torque_world = rotation @ torque_body
        p.applyExternalTorque(self.body_id, -1, torque_world.tolist(), p.WORLD_FRAME,
                              physicsClientId=self.client)
        p.stepSimulation(physicsClientId=self.client)
        self._sync_state()

    def _sync_state(self) -> None:
        position, quaternion = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client)
        linear_velocity, angular_velocity = p.getBaseVelocity(self.body_id, physicsClientId=self.client)
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(linear_velocity, dtype=float)
        self.attitude = np.asarray(p.getEulerFromQuaternion(quaternion), dtype=float)
        self.angular_velocity = np.asarray(angular_velocity, dtype=float)

    def state_vector(self) -> np.ndarray:
        """Return position, velocity, Euler attitude and angular velocity."""
        return np.concatenate((self.position, self.velocity, self.attitude, self.angular_velocity))

    def camera_image(self, width: int = 800, height: int = 600) -> np.ndarray:
        """Capture the PyBullet scene for Gymnasium's rgb_array mode."""
        view = p.computeViewMatrixFromYawPitchRoll([6.0, 4.0, 0.0], 15.0, 45.0,
                                                   -55.0, 0.0, 2)
        projection = p.computeProjectionMatrixFOV(60.0, width / height, 0.1, 50.0)
        _, _, rgba, _, _ = p.getCameraImage(width, height, view, projection,
                                             renderer=p.ER_TINY_RENDERER,
                                             physicsClientId=self.client)
        return np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

    def close(self) -> None:
        """Disconnect this environment's PyBullet client."""
        if p.isConnected(self.client):
            p.disconnect(self.client)
