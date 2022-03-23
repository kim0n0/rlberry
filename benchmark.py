import pandas as pd

from rlberry.agents.torch import PPOAgent
from rlberry.agents.torch import TRPOAgent
from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.envs.benchmarks.ball_exploration.ball2d import BallLevel1, BallLevel2, BallLevel3, BallLevel4, BallLevel5

env_dict = {0 : PBall2D(),
            1 : BallLevel1(),
            2 : BallLevel2(),
            3 : BallLevel3(),
            4 : BallLevel4(),
            5 : BallLevel5()}

idx_to_env_dict = {0 : "PBall2D",
            1 : "BallLevel1",
            2 : "BallLevel2",
            3 : "BallLevel3",
            4 : "BallLevel4",
            5 : "BallLevel5"}

def benchmark(agent_class, agent_options, n_episodes=200, horizon=256, nb_runs=100):
    for env_index in env_dict.keys():
        env = env_dict[env_index] #getting env
        agent = agent_class(
            env, horizon=horizon, **agent_options
        ) #define agent
        agent.fit(budget=n_episodes) #training 


        average_reward = 0
        for run in range(nb_runs): #getting average reward on nb_runs runs after training
            state = env.reset()
            for tt in range(horizon):
                action = agent.policy(state)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                average_reward += reward
                if done :
                    break
        
        results = pd.concat(
            [
                agent.writer.data,
                pd.DataFrame({
                    'name': [agent.name],
                    'tag': ['avg_reward'],
                    'value': [average_reward / nb_runs],
                    'global_step': [0]
                })
            ], ignore_index=True
        )
        brm = '_brm' if 'use_brm' in agent_options.keys() and agent_options['use_brm'] else ''
        filename = f"benchmarks/{agent.name}{brm}_{idx_to_env_dict[env_index]}.csv"
        results.to_csv(filename)     

ppo_options = {
    'gamma': 0.99,
    'learning_rate': 0.001,
    'eps_clip': 0.2,
    'k_epochs': 4,
    'use_brm': False
}

ppo_brm_options = {
    'gamma': 0.99,
    'learning_rate': 0.001,
    'eps_clip': 0.2,
    'k_epochs': 4,
    'use_brm': True
}

trpo_options = {
    'gamma': 0.99,
    'learning_rate': 0.05,
    'delta': 0.05,
    'num_rollouts': 50
}

benchmark(PPOAgent, ppo_options)
benchmark(TRPOAgent, trpo_options)

