import datetime

import gymnasium as gym
import panda_gym
from sb3_contrib import TQC
from stable_baselines3 import (
    DDPG,
    HerReplayBuffer,
    SAC,
    TD3,
)
from trainer import Trainer

env = gym.make(
    "PandaReach-v3",
    render_mode='human',
    # distance_threshold=0.05,
    # goal_range=0.2,
)

model = TQC(policy="MultiInputPolicy", env=env, replay_buffer_class=HerReplayBuffer, verbose=1)

trainer = Trainer(model=model, target_step=200_000)

trainer.train()
print(datetime.datetime.now())
