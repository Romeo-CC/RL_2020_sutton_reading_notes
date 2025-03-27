import numpy as np
from tqdm import tqdm

EPSILON = 1e-5 # avoid devide by zero

class Naive_Bandit:
    # Implementation of the epsilon-greedy algorithm
    def __init__(self, k_arms: int=10, epsilon: float=0., initial: float=0.):
        self.k = k_arms
        self.indices = np.arange(self.k)
        self.epsilon = epsilon
        self.initial = initial
        self.average_reward = 0
        self.true_reward = 0
        self.time = 0

    def reset(self):
        # real reward for each action
        self.q_true = np.random.randn(self.k) + self.true_reward

        # estimation for each action
        self.q_estimation = np.zeros(self.k) + self.initial

        # counter for each arm
        self.action_count = np.zeros(self.k)

        self.best_action = np.argmax(self.q_true)

        self.time = 0


    def act(self):
        # exploration
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)

        # exploitation
        q_best = np.max(self.q_estimation) 
        return np.random.choice(np.where(self.q_estimation == q_best)[0])

    # take an action, update estimation for this action
    def step(self, action):
        # generate the reward under N(real reward, 1)
        reward = np.random.randn() + self.q_true[action]
        
        self.time += 1

        self.average_reward += (reward - self.average_reward) / self.time

        self.action_count[action] += 1
        # Update estimation using sample averages
        # Incremental Implementation
        self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        
        return reward



def simulate(runs, time, bandits):
    # initial rewards and best_action_counts
    rewards = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros(rewards.shape)
    
    for i, bandit in enumerate(bandits):
        for r in tqdm(range(runs), total=runs):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
    
    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards



class Bandit(object):
    def __init__(self, k_arms: int=10, epsilon: float=0., initial: float=0., 
        step_size: float=0.1, sample_averages: bool=False, UCB_param: float=None,
        gradient: bool=False, gradient_baseline: bool=False,true_reward: float=0.):

        self.k = k_arms
        self.indices = np.arange(k_arms)
        self.epsilon = epsilon
        self.initial = initial
        self.step_size = step_size # alpha
        self.sample_averages = sample_averages
        self.UCB_param = UCB_param # c
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.true_reward = true_reward
        self.average_reward = 0
        self.time = 0
        
    
    def reset(self):
        # real reward for each action
        self.q_true = np.random.randn(self.k) + self.true_reward

        # estimation for each action
        self.q_estimation = np.zeros(self.k) + self.initial

        # counter for each arm
        self.action_count = np.zeros(self.k)

        self.best_action = np.argmax(self.q_true)

        self.time = 0


    def act(self):
        # exploration
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)

        # UCB action selection
        if self.UCB_param is not None:
            uncertainty = self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + EPSILON))
            UCB_estimation = self.q_estimation + uncertainty
            q_best = np.max(UCB_estimation)
            return np.random.choice(np.where(UCB_estimation == q_best)[0])

        # exploitation
        q_best = np.max(self.q_estimation)
        return np.random.choice(np.where(self.q_estimation == q_best)[0])

    
    def step(self, action):
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.average_reward += (reward - self.average_reward) / self.time
        self.action_count[action] += 1

        if self.sample_averages:
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]


        else:
            # exponential rencency-weighted average
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
            
        return reward
        
        
        
        
       