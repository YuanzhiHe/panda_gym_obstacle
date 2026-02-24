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
    env_id = f"Panda{task}-Obstacle"

    register(
        id=env_id,
        entry_point=f"envs.panda_tasks:Panda{task}Env",
        max_episode_steps=100 if task == "Stack" else 50,
    )

    ENV_IDS.append(env_id)

register(
    id="RoboRL-Navigator-Panda-ROS",
    entry_point="envs.robots.env_panda_ros:PandaROSEnv",
    max_episode_steps=50,
)

register(
    id="PandaReachAndPick",
    entry_point="envs.panda_tasks:PandaReachAndPickEnv",
    max_episode_steps=50,
)

register(
    id="PandaReachAndPick-Obstacle",
    entry_point="envs.panda_tasks:PandaReachAndPickObstacleEnv",
    max_episode_steps=50,
)

register(
    id="PandaPushAndPick",
    entry_point="envs.panda_tasks:PandaPushAndPickEnv",
    max_episode_steps=100,
)