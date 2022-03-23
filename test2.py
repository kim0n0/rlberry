"""
==============================================
A demo of PPO algorithm in PBall2D environment
==============================================
 Illustration of how to set up an PPO algorithm in rlberry.
 The environment chosen here is PBALL2D environment.

.. video:: ../../video_plot_ppo.mp4
   :width: 600

"""
# sphinx_gallery_thumbnail_path = 'thumbnails/video_plot_a2c.jpg'

from rlberry.agents.torch.ppo.ppo import PPOAgent
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.envs.classic_control.mountain_car import MountainCar
from rlberry.envs.classic_control.acrobot import Acrobot
from rlberry.envs.classic_control.pendulum import Pendulum
from rlberry.envs.bullet3.pybullet_envs.robot_pendula import Pendulum
from rlberry.envs.finite.gridworld import GridWorld


env = PBall2D()
# env = MountainCar()
# env = Acrobot() # ppoagent not working well with terminal states
# env = GridWorld()

n_episodes = 200
horizon = 256

agent = PPOAgent(
    env, horizon=horizon, gamma=0.99, learning_rate=0.001, k_epochs=5, use_brm=True
)

agent.fit(budget=n_episodes)

env.enable_rendering()
state = env.reset()
for tt in range(n_episodes):
    action = agent.policy(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state

video = env.save_video("_video/video_plot_ppo.mp4")
 