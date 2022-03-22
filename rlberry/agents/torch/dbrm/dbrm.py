import numpy as np
import torch
import torch.nn as nn
import logging

import gym.spaces as spaces
from rlberry.agents import AgentWithSimplePolicy
from rlberry.agents.utils.memories import Memory
from rlberry.agents.torch.utils.training import optimizer_factory
from rlberry.agents.torch.utils.models import default_policy_net_fn
from rlberry.agents.torch.utils.models import default_value_net_fn
from rlberry.utils.torch import choose_device
from rlberry.wrappers.uncertainty_estimator_wrapper import UncertaintyEstimatorWrapper


logger = logging.getLogger(__name__)


class DBRMAgent(AgentWithSimplePolicy):
    """
    Deterministic Bellman Residual Minimization Agent.

    Deterministic Bellman Residual Minimization methods for reinforcement learning,
    avoiding the usual double sampling of BRM for deterministic or middy stochastic environment

    Parameters
    ----------
    env : Model
        Online model with continuous (Box) state space and discrete actions
    batch_size : int
        Number of *episodes* to wait before updating the policy.
    horizon : int
        Horizon.
    gamma : double
        Discount factor in [0, 1].
    learning_rate : double
        Learning rate.
    optimizer_type: str
        Type of optimizer. 'ADAM' by defaut.
    k_epochs : int
        Number of epochs per update.
    policy_net_fn : function(env, **kwargs)
        Function that returns an instance of a policy network (pytorch).
        If None, a default net is used.
    value_net_fn : function(env, **kwargs)
        Function that returns an instance of a value network (pytorch).
        If None, a default net is used.
    policy_net_kwargs : dict
        kwargs for policy_net_fn
    value_net_kwargs : dict
        kwargs for value_net_fn
    device: str
        Device to put the tensors on
    use_bonus : bool, default = False
        If true, compute the environment 'exploration_bonus'
        and add it to the reward. See also UncertaintyEstimatorWrapper.
    uncertainty_estimator_kwargs : dict
        kwargs for UncertaintyEstimatorWrapper

    References
    ----------
    Schulman, J., Wolski, F., Dhariwal, P., Radford, A. & Klimov, O. (2017).
    "Proximal Policy Optimization Algorithms."
    arXiv preprint arXiv:1707.06347.

    Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015).
    "Trust region policy optimization."
    In International Conference on Machine Learning (pp. 1889-1897).
    """

    name = "DBRM"

    def __init__(
        self,
        env,
        batch_size=64,
        update_frequency=8,
        horizon=256,
        gamma=0.99,
        learning_rate=0.01,
        optimizer_type="ADAM",
        k_epochs=5,
        use_gae=True,
        gae_lambda=0.95,
        policy_net_fn=None,
        value_net_fn=None,
        policy_net_kwargs=None,
        value_net_kwargs=None,
        device="cuda:best",
        use_bonus=False,
        uncertainty_estimator_kwargs=None,
        **kwargs
    ):  # TODO: sort arguments

        AgentWithSimplePolicy.__init__(self, env, **kwargs)

        # bonus
        self.use_bonus = use_bonus
        if self.use_bonus:
            self.env = UncertaintyEstimatorWrapper(
                self.env, **uncertainty_estimator_kwargs
            )

        # algorithm parameters
        self.gamma = gamma
        self.horizon = horizon

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.k_epochs = k_epochs
        self.update_frequency = update_frequency

        # options
        # TODO: add reward normalization option
        #       add observation normalization option
        #       add orthogonal weight initialization option
        #       add value function clip option
        #       add ... ?
        self.normalize_advantages = True  # TODO: turn into argument

        self.use_gae = use_gae
        self.gae_lambda = gae_lambda

        # function approximators
        self.policy_net_kwargs = policy_net_kwargs or {}
        self.value_net_kwargs = value_net_kwargs or {}

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        #
        self.policy_net_fn = policy_net_fn or default_policy_net_fn
        self.value_net_fn = value_net_fn or default_value_net_fn

        self.device = choose_device(device)

        self.optimizer_kwargs = {"optimizer_type": optimizer_type, "lr": learning_rate}

        # check environment
        assert isinstance(self.env.observation_space, spaces.Box)
        assert isinstance(self.env.action_space, spaces.Discrete)

        self.cat_policy = None  # categorical policy function

        # initialize
        self.reset()

    @classmethod
    def from_config(cls, **kwargs):
        kwargs["policy_net_fn"] = eval(kwargs["policy_net_fn"])
        kwargs["value_net_fn"] = eval(kwargs["value_net_fn"])
        return cls(**kwargs)

    def reset(self, **kwargs):
        self.cat_policy = self.policy_net_fn(self.env, **self.policy_net_kwargs).to(
            self.device
        )
        self.policy_optimizer = optimizer_factory(
            self.cat_policy.parameters(), **self.optimizer_kwargs
        )

        self.value_net = self.value_net_fn(self.env, **self.value_net_kwargs).to(
            self.device
        )
        self.value_optimizer = optimizer_factory(
            self.value_net.parameters(), **self.optimizer_kwargs
        )

        self.cat_policy_old = self.policy_net_fn(self.env, **self.policy_net_kwargs).to(
            self.device
        )
        self.cat_policy_old.load_state_dict(self.cat_policy.state_dict())

        self.MseLoss = nn.MSELoss()  # TODO: turn into argument

        self.memory = Memory()  # TODO: Improve memory to include returns and advantages

        self.episode = 0

    def policy(self, observation):
        state = observation
        assert self.cat_policy is not None
        state = torch.from_numpy(state).float().to(self.device)
        action_dist = self.cat_policy_old(state)
        action = action_dist.sample().item()
        return action

    def fit(self, budget: int, **kwargs):
        del kwargs
        n_episodes_to_run = budget
        count = 0
        while count < n_episodes_to_run:
            self._run_episode()
            count += 1

    def _run_episode(self):
        # to store transitions
        states = []
        next_states = []
        rewards = []
        is_terminals = []

        # interact for H steps
        episode_rewards = 0
        state = self.env.reset()
        state = torch.from_numpy(state).float().to(self.device)

        for _ in range(self.horizon):
            # running policy_old
            action_dist = self.cat_policy_old(state)
            action = action_dist.sample()
            next_state, reward, done, info = self.env.step(action.item())
            next_state = torch.from_numpy(next_state).float().to(self.device)

            # check whether to use bonus
            bonus = 0.0
            if self.use_bonus:
                if info is not None and "exploration_bonus" in info:
                    bonus = info["exploration_bonus"]

            # save transition
            states.append(state)
            next_states.append(next_state)
            rewards.append(reward + bonus)  # bonus added here
            is_terminals.append(done)

            episode_rewards += reward

            if done:
                break

            # update state
            state = next_state

        # compute returns and advantages
        state_values = self.value_net(torch.stack(states).to(self.device)).detach()
        state_values = torch.squeeze(state_values).tolist()

        rewards = torch.Tensor(rewards).float().to(self.device)

        # save in batch
        self.memory.states.extend(states)
        self.memory.next_states.extend(next_states)
        self.memory.rewards.extend(rewards)
        self.memory.is_terminals.extend(is_terminals)

        # increment ep counter
        self.episode += 1

        # log
        if self.writer is not None:
            self.writer.add_scalar("episode_rewards", episode_rewards, self.episode)

        # update
        if (
            self.episode % self.update_frequency == 0
        ):  # TODO: maybe change to update in function of n_steps instead
            self._update()
            self.memory.clear_memory()

        return episode_rewards

    def _update(self):
        # convert list to tensor
        full_old_states = torch.stack(self.memory.states).to(self.device).detach()
        full_new_states = torch.stack(self.memory.next_states).to(self.device).detach()
        full_old_rewards = torch.stack(self.memory.rewards).to(self.device).detach()

        # optimize policy for K epochs
        n_samples = full_old_states.size(0)
        n_batches = n_samples // self.batch_size

        for _ in range(self.k_epochs):

            # shuffle samples
            rd_indices = self.rng.choice(n_samples, size=n_samples, replace=False)
            shuffled_states = full_old_states[rd_indices]
            shuffled_next_states = full_new_states[rd_indices]
            shuffled_rewards = full_old_rewards[rd_indices]

            for k in range(n_batches):

                # sample batch
                batch_idx = np.arange(
                    k * self.batch_size, min((k + 1) * self.batch_size, n_samples)
                )
                old_states = shuffled_states[batch_idx]
                new_states = shuffled_next_states[batch_idx]
                old_rewards = shuffled_rewards[batch_idx]

                # evaluate old actions and values
                state_values = torch.squeeze(self.value_net(old_states))
                new_state_values = torch.squeeze(self.value_net(new_states))

                # compute total loss
                # loss = -surr_loss + loss_vf - loss_entropy
                loss = self.MseLoss(state_values, old_rewards + self.gamma*new_state_values)

                # take gradient step
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()

                loss.mean().backward()

                self.policy_optimizer.step()
                self.value_optimizer.step()

        # log
        if self.writer:
            self.writer.add_scalar(
                "fit/brm_loss",
                loss.mean().cpu().detach().numpy(),
                self.episode,
            )

        # copy new weights into old policy
        self.cat_policy_old.load_state_dict(self.cat_policy.state_dict())

    #
    # For hyperparameter optimization
    #
    @classmethod
    def sample_parameters(cls, trial):
        batch_size = trial.suggest_categorical("batch_size", [1, 4, 8, 16, 32])
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.99])
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
        k_epochs = trial.suggest_categorical("k_epochs", [1, 5, 10, 20])

        return {
            "batch_size": batch_size,
            "gamma": gamma,
            "learning_rate": learning_rate,
            "k_epochs": k_epochs,
        }
