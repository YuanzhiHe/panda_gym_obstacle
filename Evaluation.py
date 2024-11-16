import imageio
import gymnasium as gym
import numpy as np
from sb3_contrib import TQC
import envs

env = gym.make(
    "PandaPushAndPick",
    # render_mode="human",    # do not use human mode if generating gif image
    obs_num=0
)

model = TQC.load("/home/yuanzhihe/PycharmProjects/csv2matplotlib/models/Pick2PushAndPick3/model", env=env)

# obs, _ = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, _, info = env.step(action)
#     env.render()


# generate gif image

images = []
obs, _ = env.reset()
img = env.render()
for i in range(500):
    images.append(img)
    action, _ = model.predict(obs)
    obs, _, _, _, _ = env.step(action)
    img = env.render()

imageio.mimsave('Pick2PushAndPick.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)