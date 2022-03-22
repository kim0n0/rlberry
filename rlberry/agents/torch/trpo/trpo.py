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


from collections import namedtuple
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam

logger = logging.getLogger(__name__)

def surrogate_loss(new_probabilities, old_probabilities, advantages):
    return (new_probabilities / old_probabilities * advantages).mean()


def kl_div(p, q):
    p = p.detach()
    return (p * (p.log() - q.log())).sum(-1).mean()

def flat_grad(y, x, retain_graph=False, create_graph=False):
        if create_graph:
            retain_graph = True

        g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
        g = torch.cat([t.view(-1) for t in g])
        return g

def conjugate_gradient(A, b, delta=0.1, max_iterations=float('inf')):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()

    i = 0
    while i < max_iterations:
        AVP = A(p)

        dot_old = r @ r
        alpha = dot_old / (p @ AVP)

        x_new = x + alpha * p

        if (x - x_new).norm() <= delta:
            return x_new

        i += 1
        r = r - alpha * AVP

        beta = (r @ r) / dot_old
        p = r + beta * p

        x = x_new
    return x

class TRPOAgent(AgentWithSimplePolicy):
    """
    TRPO Agent

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

    name = "TRPO"

    def __init__(
        self,
        env,
        batch_size=64,
        horizon=256,
        gamma=0.99,
        learning_rate=0.01,
        policy_net_fn=None,
        value_net_fn=None,
        policy_net_kwargs=None,
        value_net_kwargs=None,
        device="cuda:best",
        use_bonus=False,
        uncertainty_estimator_kwargs=None,
        delta = 0.1,
        num_rollouts = 100,
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


        # options
        # TODO: add reward normalization option
        #       add observation normalization option
        #       add orthogonal weight initialization option
        #       add value function clip option
        #       add ... ?
        self.normalize_advantages = True  # TODO: turn into argument

        # function approximators
        self.policy_net_kwargs = policy_net_kwargs or {}
        self.value_net_kwargs = value_net_kwargs or {}

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        #
        self.policy_net_fn = policy_net_fn or default_policy_net_fn
        self.value_net_fn = value_net_fn or default_value_net_fn

        self.device = choose_device(device)

        # check environment
        assert isinstance(self.env.observation_space, spaces.Box)
        assert isinstance(self.env.action_space, spaces.Discrete)

        # initialize
        self.reset()


        self.delta = delta
        self.num_rollouts = num_rollouts

        self.Rollout = namedtuple('Rollout', ['states', 'actions', 'rewards', 'next_states', ])


        

    @classmethod
    def from_config(cls, **kwargs):
        kwargs["policy_net_fn"] = eval(kwargs["policy_net_fn"])
        kwargs["value_net_fn"] = eval(kwargs["value_net_fn"])
        return cls(**kwargs)

    def reset(self, **kwargs):
        actor_hidden = 32
        self.actor = nn.Sequential(nn.Linear(self.state_dim, actor_hidden),
                            nn.ReLU(),
                            nn.Linear(actor_hidden, self.action_dim),
                            nn.Softmax(dim=1))

        critic_hidden = 32
        self.critic = nn.Sequential(nn.Linear(self.state_dim, critic_hidden),
                            nn.ReLU(),
                            nn.Linear(critic_hidden, 1))
        
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.learning_rate)

        self.episode = 0

    def policy(self, observation):
        state = observation
        state = torch.tensor(state).float().unsqueeze(0)  # Turn state into a batch with a single element
        dist = Categorical(self.actor(state))  # Create a distribution from probabilities for actions
        return dist.sample().item()

    def fit(self, budget: int, **kwargs):
        del kwargs
        n_episodes_to_run = budget
        count = 0
        while count < n_episodes_to_run:
            self._run_episode()
            count += 1

    def _run_episode(self):
        rollouts = []

        
        episode_rewards = 0

        for t in range(self.num_rollouts):
            state = self.env.reset()
            done = False

            samples = []


            for _ in range(self.horizon):
                with torch.no_grad():
                    action = self.policy(state)
                    next_state, reward, done, _ = self.env.step(action)

                    # Collect samples
                    samples.append((state, action, reward, next_state))

                    state = next_state

                    episode_rewards += reward
                    if done :
                        break

            # Transpose our samples
            states, actions, rewards, next_states = zip(*samples)

            states = torch.stack([torch.from_numpy(state) for state in states], dim=0).float()
            next_states = torch.stack([torch.from_numpy(state) for state in next_states], dim=0).float()
            actions = torch.as_tensor(actions).unsqueeze(1)
            rewards = torch.as_tensor(rewards).unsqueeze(1)

            rollouts.append(self.Rollout(states, actions, rewards, next_states))

        self._update(rollouts)
        self.episode += 1
        # log
        if self.writer is not None:
            self.writer.add_scalar("episode_rewards", episode_rewards/self.num_rollouts, self.episode)

    def update_critic(self, advantages):
        loss = .5 * (advantages ** 2).mean()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()


    def estimate_advantages(self, states, last_state, rewards):
        values = self.critic(states)
        last_value = self.critic(last_state.unsqueeze(0))
        
        next_values = torch.zeros_like(rewards)
        for i in reversed(range(rewards.shape[0])):
            last_value = next_values[i] = rewards[i] + 0.99 * last_value
            
        advantages = next_values - values
        return advantages

    def apply_update(self, grad_flattened):
        n = 0
        for p in self.actor.parameters():
            numel = p.numel()
            g = grad_flattened[n:n + numel].view(p.shape)
            p.data += g
            n += numel



    def _update(self, rollouts):

        states = torch.cat([r.states for r in rollouts], dim=0)
        actions = torch.cat([r.actions for r in rollouts], dim=0).flatten()

        advantages = [self.estimate_advantages(states, next_states[-1], rewards) for states, _, rewards, next_states in rollouts]
        advantages = torch.cat(advantages, dim=0).flatten()

        self.update_critic(advantages)

        distribution = self.actor(states)
        distribution = torch.distributions.utils.clamp_probs(distribution)
        probabilities = distribution[range(distribution.shape[0]), actions]

        # We will calculate the gradient wrt to the new probabilities (surrogate function),
        # so second probabilities should be treated as a constant
        L = surrogate_loss(probabilities, probabilities.detach(), advantages)
        KL = kl_div(distribution, distribution)

        parameters = list(self.actor.parameters())

        g = flat_grad(L, parameters, retain_graph=True)  # We will use the graph several times
        d_kl = flat_grad(KL, parameters, create_graph=True)  # Create graph, because we will call backward() on it (for HVP)

        def HVP(v):
            return flat_grad(d_kl @ v, parameters, retain_graph=True)

        search_dir = conjugate_gradient(HVP, g, self.delta, max_iterations=20)
        max_length = torch.sqrt(2 * self.delta / (search_dir @ HVP(search_dir)))
        max_step = max_length * search_dir

        def criterion(step):
            # Apply parameters' update
            self.apply_update(step)

            with torch.no_grad():
                distribution_new = self.actor(states)
                distribution_new = torch.distributions.utils.clamp_probs(distribution_new)
                probabilities_new = distribution_new[range(distribution_new.shape[0]), actions]

                L_new = surrogate_loss(probabilities_new, probabilities, advantages)
                KL_new = kl_div(distribution, distribution_new)

            L_improvement = L_new - L
            if L_improvement > 0 and KL_new <= self.delta:
                return True

            # Step size too big, reverse
            self.apply_update(-step)
            return False

        i = 0
        while not criterion((0.9 ** i) * max_step) and i < 10:
            i += 1

