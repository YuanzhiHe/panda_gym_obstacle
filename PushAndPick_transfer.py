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

# from tqc_panda import TQC_panda
from trainer import Trainer
import envs

env = gym.make(
    "PandaPushAndPick",
    render_mode="rgb_array",
    obs_num=0
)
env1 = gym.make(
    "PandaPickAndPlace-v3",
    render_mode="rgb_array",
)
env2 = gym.make(
    "PandaPush-v3",
    render_mode="rgb_array",
)
env3 = gym.make(
    "PandaReach-v3",
    render_mode="rgb_array",
)

# PC use
# model2 = TQC.load("/home/yuanzhihe/PycharmProjects/csv2matplotlib/models/PandaReach/model", env=env3)
# model1 = TQC.load("/home/yuanzhihe/PycharmProjects/csv2matplotlib/models/PandaPick/model", env=env1)
# model = TQC.load("/home/yuanzhihe/PycharmProjects/csv2matplotlib/models/PandaPush2/model", env=env2)
# supercomputer use
model2 = TQC.load("/scratch/c.c23089110/panda_gym_obstacle/models/PandaReach/model", env=env3)
model1 = TQC.load("/scratch/c.c23089110/panda_gym_obstacle/models/PandaPick/model", env=env1)
model = TQC.load("/scratch/c.c23089110/panda_gym_obstacle/models/PandaPush2/model", env=env2)

model3 = TQC(policy="MultiInputPolicy",
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
            learning_rate=0.0001,
            verbose=1,
            policy_kwargs=dict(net_arch=[1024, 1024, 1024], n_critics=2),
            device="cuda"
)

push_first_layer = [model.critic.q_networks[0][0], model.critic.q_networks[1][0]]
pick_first_layer = [model1.critic.q_networks[0][0], model1.critic.q_networks[1][0]]
reach_first_layer = [model2.critic.q_networks[0][0], model2.critic.q_networks[1][0]]
push_rest_layer = [model.critic.q_networks[0][1:], model.critic.q_networks[1][1:]]
pick_rest_layer = [model1.critic.q_networks[0][1:], model1.critic.q_networks[1][1:]]

model3.critic.q_networks[0][0].weight.data[:512, :6] = reach_first_layer[0].weight.data[:, :6]
model3.critic.q_networks[0][0].weight.data[:512, 6:19] = pick_first_layer[0].weight.data[:, 6:19]
model3.critic.q_networks[0][0].weight.data[512:, 19:40] = push_first_layer[0].weight.data[:, 6:27]
model3.critic.q_networks[0][0].weight.data[:512, 40] = pick_first_layer[0].weight.data[:, 28]
model3.critic.q_networks[1][0].weight.data[:512, :6] = reach_first_layer[1].weight.data[:, :6]
model3.critic.q_networks[1][0].weight.data[:512, 6:19] = pick_first_layer[1].weight.data[:, 6:19]
model3.critic.q_networks[1][0].weight.data[512:, 19:40] = push_first_layer[1].weight.data[:, 6:27]
model3.critic.q_networks[1][0].weight.data[:512, 40] = pick_first_layer[1].weight.data[:, 28]

layer_num = [2,4,6]

for i in layer_num:
    if i == 6:
        model3.critic.q_networks[0][i].weight.data[:, :512] = pick_rest_layer[0][i-1].weight.data
        model3.critic.q_networks[1][i].weight.data[:, :512] = pick_rest_layer[1][i-1].weight.data
        model3.critic.q_networks[0][i].weight.data[:, 512:] = push_rest_layer[0][i-1].weight.data
        model3.critic.q_networks[1][i].weight.data[:, 512:] = push_rest_layer[1][i-1].weight.data
        model3.critic.q_networks[0][i].bias.data = push_rest_layer[0][i-1].bias.data
        model3.critic.q_networks[1][i].bias.data = push_rest_layer[1][i-1].bias.data
    else:
        model3.critic.q_networks[0][i].weight.data[:512, :512] = pick_rest_layer[0][i-1].weight.data
        model3.critic.q_networks[1][i].weight.data[:512, :512] = pick_rest_layer[1][i-1].weight.data
        model3.critic.q_networks[0][i].weight.data[512:, 512:] = push_rest_layer[0][i-1].weight.data
        model3.critic.q_networks[1][i].weight.data[512:, 512:] = push_rest_layer[1][i-1].weight.data
        model3.critic.q_networks[0][i].bias.data[:512] = pick_rest_layer[0][i-1].bias.data
        model3.critic.q_networks[1][i].bias.data[:512] = pick_rest_layer[1][i-1].bias.data
        model3.critic.q_networks[0][i].bias.data[512:] = push_rest_layer[0][i-1].bias.data
        model3.critic.q_networks[1][i].bias.data[512:] = push_rest_layer[1][i-1].bias.data


trainer = Trainer(model=model3, target_step=1000_000)

trainer.train()
print(datetime.datetime.now())
