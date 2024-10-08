from typing import Any, Dict

import numpy as np

from envs.core import Task
from utils import distance


class Reach(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_range=0.3,
        obs_num: int = 3
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.obstacle1_pos = None
        self.obstacle2_pos = None
        self.obstacle3_pos = None
        self.get_ee_position = get_ee_position
        self.obs_num = obs_num
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        self.obstacle_range = goal_range
        self.obstacle_range_low = np.array([0.2 - self.obstacle_range, -self.obstacle_range, 0.05])
        self.obstacle_range_high = np.array([0.2 + self.obstacle_range, self.obstacle_range, 0.05])
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.3, width=2, height=0.4)
        self.sim.create_obstacles(length=0.05, width=0.05, height=0.05, obs_num=self.obs_num)
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        obstacle_position = self.obstacle1_pos
        observation = np.concatenate([obstacle_position])
        if self.obs_num > 1:
            obstacle_position = self.obstacle2_pos
            observation = np.concatenate([observation, obstacle_position])
        if self.obs_num == 3:
            obstacle_position = self.obstacle3_pos
            observation = np.concatenate([observation, obstacle_position])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        self.goal = self._sample_goal()

        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

        if self.obs_num == 1:
            self.obstacle1_pos = self._sample_obstacles()
            self.sim.set_base_pose("obstacle1", self.obstacle1_pos, np.array([0.0, 0.0, 0.0, 1.0]))
        elif self.obs_num == 2:
            self.obstacle1_pos, self.obstacle2_pos = self._sample_obstacles()
            self.sim.set_base_pose("obstacle1", self.obstacle1_pos, np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("obstacle2", self.obstacle2_pos, np.array([0.0, 0.0, 0.0, 1.0]))
        else:
            self.obstacle1_pos, self.obstacle2_pos, self.obstacle3_pos = self._sample_obstacles()
            self.sim.set_base_pose("obstacle1", self.obstacle1_pos, np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("obstacle2", self.obstacle2_pos, np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("obstacle3", self.obstacle3_pos, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal

    def _sample_obstacles(self):
        min_dist_to_goal = 0.1

        position1 = np.random.uniform(self.obstacle_range_low, self.obstacle_range_high)
        while distance(position1, self.goal) < min_dist_to_goal:
            position1 = np.random.uniform(self.obstacle_range_low, self.obstacle_range_high)
        if self.obs_num == 2:
            position2 = np.random.uniform(self.obstacle_range_low, self.obstacle_range_high)
            while distance(position2, self.goal) < min_dist_to_goal:
                position2 = np.random.uniform(self.obstacle_range_low, self.obstacle_range_high)
            return position1, position2
        elif self.obs_num == 3:
            position2 = np.random.uniform(self.obstacle_range_low, self.obstacle_range_high)
            while distance(position2, self.goal) < min_dist_to_goal:
                position2 = np.random.uniform(self.obstacle_range_low, self.obstacle_range_high)
            position3 = np.random.uniform(self.obstacle_range_low, self.obstacle_range_high)
            while distance(position3, self.goal) < min_dist_to_goal:
                position3 = np.random.uniform(self.obstacle_range_low, self.obstacle_range_high)
            return position1, position2, position3
        else:
            return position1

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)
