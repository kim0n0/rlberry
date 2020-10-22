import numpy as np
from rlberry.agents.dynprog.utils import backward_induction, value_iteration
from rlberry.agents         import Agent
from rlberry.agents.dynprog import ValueIterationAgent
from rlberry.envs.interface import GenerativeModel 
from rlberry.envs.finite    import FiniteMDP
from rlberry.spaces         import Discrete


class MBQVIAgent(Agent):
    """
    Model-Basel Q-Value iteration (MBQVI).

    Builds an empirical MDP and runs value iteration on it.
    Corresponds to the "indirect" algorithm studied by Kearns and Singh (1999).

    Kearns, Michael J., and Satinder P. Singh. 
    "Finite-sample convergence rates for Q-learning and indirect algorithms." 
    Advances in neural information processing systems. 1999.
    """
    def __init__(self, env, 
                       n_samples=10, 
                       gamma=0.95, 
                       horizon=None, 
                       epsilon=1e-6, 
                       verbose=1, 
                       **kwargs):
        """
        Parameters:
        -----------
        env : GenerativeModel
            generative model with finite state-action space
        n_samples : int 
            number of samples *per state-action pair* used to estimate the empirical MDP
        gamma : double 
            discount factor in [0, 1]
        horizon : int
            horizon, if the problem is finite-horizon. if None, the discounted problem is solved
            default = None
        epsilon : double
            precision of value iteration, only used in discounted problems (when horizon is None).
        verbose : int
            controls the verbosity, if non zero, progress messages are printed.
        """
        # initialize base class
        assert isinstance(env, GenerativeModel), "MBQVI requires a generative model."
        assert isinstance(env.observation_space, Discrete), "MBQVI requires a finite state space."
        assert isinstance(env.action_space,      Discrete), "MBQVI requires a finite action space."
        Agent.__init__(self, env)
        self.id = "MBQVI"

        # 
        self.n_samples = n_samples
        self.gamma = gamma
        self.horizon = horizon 
        self.epsilon = epsilon
        self.verbose = verbose

        # empirical MDP, created in fit()
        self.R_hat = None 
        self.P_hat = None 

        # value functions
        self.V = None 
        self.Q = None

    def _update(self, state, action, next_state, reward):
        """
        Update model statistics.
        """
        self.N_sa[state, action] += 1
        self.N_sas[state, action, next_state] += 1
        self.S_sa[state, action] += reward

    def fit(self, **kwargs):
        """
        Build empirical MDP and run value iteration.
        """
        S = self.env.observation_space.n 
        A = self.env.action_space.n 
        self.N_sa  = np.zeros((S, A))
        self.N_sas = np.zeros((S, A, S))
        self.S_sa  = np.zeros((S, A))

        # collect data
        total_samples = S*A*self.n_samples
        count = 0 
        if self.verbose > 0:
            print("[%s] collecting %d samples per (s,a), total = %d samples"%(self.id, self.n_samples, total_samples))
        for ss in range(S):
            for aa in range(A):
                for nn in range(self.n_samples):                 
                    next_state, reward, _, _ = self.env.sample(ss, aa)
                    self._update(ss, aa, next_state, reward)

                    count += 1
                    if count % 10000 == 0 and self.verbose > 0:
                        completed = 100*count/total_samples
                        print("[%s] ... %d/%d  (%0.0f"%(self.id, count, total_samples, completed)+"%)")

        # build model and run VI
        if self.verbose > 0:
            print("[%s] building model and running backward induction..."%self.id)

        N_sa  = np.maximum(self.N_sa, 1)
        self.R_hat = self.S_sa/N_sa 
        self.P_hat = np.zeros((S, A, S))
        for ss in range(S):
            self.P_hat[:, :, ss] = self.N_sas[:,:,ss]/N_sa 

        info = {}
        info["n_samples"] = self.n_samples
        info["total_samples"] = total_samples
        if self.horizon is None:
            assert self.gamma < 1.0, "The discounted setting requires gamma < 1.0"
            self.Q, self.V, n_it = value_iteration(self.R_hat, self.P_hat, self.gamma, self.epsilon)
            info["n_iterations"] = n_it
            info["precision"] = self.epsilon
        else:
            self.Q, self.V = backward_induction(self.R_hat, self.P_hat, self.horizon, self.gamma)
            info["n_iterations"] = self.horizon
        return info

    def policy(self, state, hh=0, **kwargs):
        """
        Parameters
        -----------
        state : int 
        hh : int
            stage when action is taken (for finite horizon problems, the optimal policy depends on hh)
            not used if horizon is None.
        """
        assert self.env.observation_space.contains(state)
        if self.horizon is None:
            return self.Q[state, :].argmax()
        else:
            assert hh >= 0 and hh < self.horizon
            return self.Q[hh, state, :].argmax()
