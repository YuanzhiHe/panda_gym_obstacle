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
env2 = gym.make(
    "PandaReach-Obstacle",
    render_mode='rgb_array',
    distance_threshold=0.05,
    obs_num=1
)
# PC use
# model = TQC.load("/home/yuanzhihe/PycharmProjects/csv2matplotlib/models/PandaReachObstacle2_1/model", env=env2)

# supercomputer use
model = TQC.load("/scratch/c.c23089110/panda_gym_obstacle/models/PandaReachObstacle/model", env=env2)

model2 = TQC(policy="MultiInputPolicy",
            env=env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy='future',
            ),
            tau=0.05,
            buffer_size=1000000,
            batch_size=256,
            gamma=0.95,
            learning_rate=0.001,
            verbose=1,
            policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
)

original_first_layer = [model.critic.q_networks[0][0], model.critic.q_networks[1][0]]
model2.critic.q_networks[0][0].weight.data[:,:7] = original_first_layer[0].weight.data[:,:7]
model2.critic.q_networks[0][0].weight.data[:,20:23] = original_first_layer[0].weight.data[:,7:10]
model2.critic.q_networks[0][0].weight.data[:,24:] = original_first_layer[0].weight.data[:,10:]
model2.critic.q_networks[1][0].weight.data[:,:7] = original_first_layer[1].weight.data[:,:7]
model2.critic.q_networks[1][0].weight.data[:,20:23] = original_first_layer[1].weight.data[:,7:10]
model2.critic.q_networks[1][0].weight.data[:,24:] = original_first_layer[1].weight.data[:,10:]

model2.critic.q_networks[0][0].bias.data = original_first_layer[0].bias.data
model2.critic.q_networks[1][0].bias.data = original_first_layer[1].bias.data

trainer = Trainer(model=model2, target_step=2000_000)

trainer.train()
print(datetime.datetime.now())
