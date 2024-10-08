from typing import Any, Dict

import numpy as np

from envs.core import Task
from pybulletTest import PyBulletTest
from utils import distance


class PickAndPlace(Task):
    def __init__(
        self,
        sim: PyBulletTest,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.3,
        goal_x_offset=0.4,
        goal_z_range: float = 0.2,
        obj_xy_range: float = 0.3,
        obs_num: int = 3
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.obstacle1_pos = None
        self.obstacle2_pos = None
        self.obstacle3_pos = None
        self.object_size = 0.04
        self.obs_num = obs_num
        self.goal_range_low = np.array([-goal_xy_range / 2 + goal_x_offset, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2 + goal_x_offset, goal_xy_range / 2, goal_z_range])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        self.obstacle_range = goal_xy_range
        self.obstacle_range_low = np.array([0.2 - self.obstacle_range, -self.obstacle_range, 0.05])
        self.obstacle_range_high = np.array([0.2 + self.obstacle_range, self.obstacle_range, 0.05])
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.3, width=2, height=0.4)
        self.sim.create_obstacles(length=0.05, width=0.05, height=0.05, obs_num=self.obs_num)
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        obstacle_position = self.obstacle1_pos
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity, obstacle_position])
        if self.obs_num > 1:
            obstacle_position = self.obstacle2_pos
            observation = np.concatenate([observation, obstacle_position])
        if self.obs_num == 3:
            obstacle_position = self.obstacle3_pos
            observation = np.concatenate([observation, obstacle_position])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        object_position = self._sample_object()

        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))

        if self.obs_num == 1:
            self.obstacle1_pos = self._sample_obstacles(object_position)
            self.sim.set_base_pose("obstacle1", self.obstacle1_pos, np.array([0.0, 0.0, 0.0, 1.0]))
        elif self.obs_num == 2:
            self.obstacle1_pos, self.obstacle2_pos = self._sample_obstacles(object_position)
            self.sim.set_base_pose("obstacle1", self.obstacle1_pos, np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("obstacle2", self.obstacle2_pos, np.array([0.0, 0.0, 0.0, 1.0]))
        else:
            self.obstacle1_pos, self.obstacle2_pos, self.obstacle3_pos = self._sample_obstacles(object_position)
            self.sim.set_base_pose("obstacle1", self.obstacle1_pos, np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("obstacle2", self.obstacle2_pos, np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("obstacle3", self.obstacle3_pos, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        goal = np.array([0.0, 0.0, self.object_size / 2])  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        if self.np_random.random() < 0.3:
            noise[2] = 0.0
        goal += noise
        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def _sample_obstacles(self, object_position):
        min_dist_to_goal = 0.1

        position1 = np.random.uniform(self.obstacle_range_low, self.obstacle_range_high)
        while distance(position1, self.goal) < min_dist_to_goal or distance(position1, object_position) < min_dist_to_goal:
            position1 = np.random.uniform(self.obstacle_range_low, self.obstacle_range_high)
        if self.obs_num == 2:
            position2 = np.random.uniform(self.obstacle_range_low, self.obstacle_range_high)
            while distance(position2, self.goal) < min_dist_to_goal or distance(position2, object_position) < min_dist_to_goal:
                position2 = np.random.uniform(self.obstacle_range_low, self.obstacle_range_high)
            return position1, position2
        elif self.obs_num == 3:
            position2 = np.random.uniform(self.obstacle_range_low, self.obstacle_range_high)
            while distance(position2, self.goal) < min_dist_to_goal or distance(position2, object_position) < min_dist_to_goal:
                position2 = np.random.uniform(self.obstacle_range_low, self.obstacle_range_high)
            position3 = np.random.uniform(self.obstacle_range_low, self.obstacle_range_high)
            while distance(position3, self.goal) < min_dist_to_goal or distance(position3, object_position) < min_dist_to_goal:
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
