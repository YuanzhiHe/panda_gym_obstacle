from typing import (
    Dict,
    Any,
    Optional,
    Tuple,
)

import numpy as np

from gymnasium.utils import seeding

from envs.core import RobotTaskEnv
from envs.robots.ros_sim import ROSSim
from envs.robots.ros_panda_robot import ROSRobot
from envs.tasks.reach import Reach


class PandaROSEnv(RobotTaskEnv):

    def __init__(
        self,
        orientation_task: bool = False,
        distance_threshold: float = 0.05,
        goal_range: float = 0.3,
        demonstration: bool = False,
        real_robot: bool = False,
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
        obs_num: int = 3
    ) -> None:
        sim = ROSSim(orientation_task=orientation_task, demonstration=demonstration)
        robot = ROSRobot(sim, orientation_task=orientation_task, real_robot=real_robot)
        task = Reach(
            sim,
            robot,
            reward_type="dense",
            orientation_task=orientation_task,
            distance_threshold=distance_threshold,
            demonstration=demonstration,
            get_ee_position=robot.get_ee_position,
            goal_range=goal_range,
            obs_num=obs_num
        )
        self.demonstration = demonstration
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if self.demonstration and options and "goal" in options:
            self.task.set_goal(options["goal"])
        else:
            super().reset(seed=seed, options=options)
            self.task.np_random, seed = seeding.np_random(seed)
            with self.sim.no_rendering():
                self.robot.reset()
                self.task.reset()

        observation = self._get_obs()
        info = {"is_success": self.task.is_success(observation["achieved_goal"], self.task.get_goal())}
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        truncated = self.robot.set_action(action)
        observation = self._get_obs()
        terminated = bool(self.task.is_success(observation["achieved_goal"], self.task.get_goal()))
        info = {"is_success": terminated}
        reward = float(self.task.compute_reward(observation["achieved_goal"], self.task.get_goal(), info))
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        pass

    def render(self) -> Optional[np.ndarray]:
        pass
