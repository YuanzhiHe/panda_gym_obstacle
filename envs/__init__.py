from envs.panda_tasks import (
    PandaFlipEnv,
    PandaPickAndPlaceEnv,
    PandaPushEnv,
    PandaReachEnv,
    PandaSlideEnv,
    PandaStackEnv,
)

import os

from gymnasium.envs.registration import register

with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

ENV_IDS = []

for task in ["Reach", "Slide", "Push", "PickAndPlace", "Stack", "Flip"]:
    for reward_type in ["sparse", "dense"]:
        for control_type in ["ee", "joints"]:
            reward_suffix = "Dense" if reward_type == "dense" else ""
            control_suffix = "Joints" if control_type == "joints" else ""
            env_id = f"Panda{task}{control_suffix}{reward_suffix}-Obstacle"

            register(
                id=env_id,
                entry_point=f"envs.panda_tasks:Panda{task}Env",
                # kwargs={"reward_type": reward_type, "control_type": control_type, "distance_threshold": distance_threshold},
                max_episode_steps=100 if task == "Stack" else 50,
            )

            ENV_IDS.append(env_id)

register(
    id="RoboRL-Navigator-Panda-ROS",
    entry_point="envs.robots.env_panda_ros:PandaROSEnv",
    max_episode_steps=50,
)