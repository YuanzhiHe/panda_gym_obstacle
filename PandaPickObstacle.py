import datetime

import gymnasium as gym
# import panda_gym
from sb3_contrib import TQC
from stable_baselines3 import (
    DDPG,
    HerReplayBuffer,
    SAC,
    TD3,
)
from trainer import Trainer
import envs

env = gym.make(
    "PandaPickAndPlace-Obstacle",
    render_mode="rgb_array",
    distance_threshold=0.05,
    obs_num=1
)

model = TD3(policy="MultiInputPolicy",
            env=env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy='future',
            ),
            tau=0.05,
            buffer_size=1000000,
            batch_size=2048,
            gamma=0.95,
            learning_rate=0.001,
            verbose=1,
            policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
            device="cuda"
)

trainer = Trainer(model=model, target_step=4000_000)

trainer.train()
print(datetime.datetime.now())
