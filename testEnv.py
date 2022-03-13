# %%
from rlberry.envs import Acrobot
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.envs.benchmarks.generalization.twinrooms import TwinRooms
from rlberry.envs.benchmarks.grid_exploration.apple_gold import AppleGold
from rlberry.envs.benchmarks.grid_exploration.nroom import NRoom
from rlberry.envs.classic_control import MountainCar
from rlberry.envs.finite import Chain, GridWorld
from gym.envs.classic_control import CartPoleEnv
from rlberry.agents.agent import Agent
from time import time

ALL_ENVS = {
    'Acrobot': Acrobot,
    'PBall2D': PBall2D,
    'TwinRooms': TwinRooms,
    'AppleGold': AppleGold,
    'NRoom': NRoom,
    'MountainCar': MountainCar,
    'Chain': Chain,
    'GridWorld': GridWorld,
    'CartPoleEnv': CartPoleEnv
}


def make_env(name: str, **kwargs):
    if name not in ALL_ENVS.keys():
        raise ValueError(f'{name} not in {ALL_ENVS.keys()}')
    return ALL_ENVS[name](**kwargs)


BASE_BENCHMARKS = ['MountainCar', 'Acrobot', 'CartPoleEnv']


def base_benchmarks(model):
    for bench in BASE_BENCHMARKS:
        results = benchmark(bench, model)


def benchmark(bench, agent_class: Agent):
    agent = agent_class(env=bench)
    start = time()
    agent.fit()
    training_time = time()-start
    performance = agent.eval()
    return training_time, performance

    # %%
