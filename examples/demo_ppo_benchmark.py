import rlberry.seeding as seeding
from rlberry.envs.benchmarks.ball_exploration.ball2d import get_benchmark_env
from rlberry.agents import MBQVIAgent
from rlberry.agents.ppo import PPOAgent
from rlberry.wrappers import RescaleRewardWrapper, DiscretizeStateWrapper
from rlberry.eval.agent_stats import AgentStats, plot_episode_rewards, compare_policies


# global seed
seeding.set_global_seed(1234)

# --------------------------------
# Define train and evaluation envs
# --------------------------------
train_env = get_benchmark_env(level=5)
d_train_env = DiscretizeStateWrapper(train_env, 20) 


# -----------------------------
# Parameters
# -----------------------------
N_EPISODES = 500
GAMMA = 0.99
HORIZON = 50
VERBOSE = 4

params_oracle = {
    "n_samples": 20,  # samples per state-action
    "gamma":   GAMMA,
    "horizon": HORIZON
}

params_ppo = {"n_episodes" : N_EPISODES,
              "gamma" : GAMMA,
              "horizon" : HORIZON,
              "learning_rate": 0.0003,
              "verbose":VERBOSE}

# -----------------------------
# Run AgentStats
# -----------------------------
oracle_stats = AgentStats(MBQVIAgent, d_train_env, init_kwargs=params_oracle, nfit=4, agent_name="Oracle")
ppo_stats    = AgentStats(PPOAgent,   train_env,   init_kwargs=params_ppo,    nfit=4, agent_name="PPO")

agent_stats_list = [oracle_stats, ppo_stats]

# learning curves
plot_episode_rewards(agent_stats_list, cumulative=True, show=False)

# compare final policies
output = compare_policies(agent_stats_list, eval_horizon=HORIZON, nsim=10)
print(output)