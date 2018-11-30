import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
from a2c.storage import RolloutStorage
from a2c.model import Actor, Critic
from a2c.rollout import RolloutCollector
import gym
import time


def get_env_info(env_name):
    env = gym.make(env_name)
    return env.observation_space, env.action_space


class A2C:

    def __init__(self,
                 env_name="BipedalWalker-v2",
                 num_steps=5,
                 num_workers=10,
                 num_updates=10000,
                 log_frequency=10,
                 use_gae=True,
                 gamma=0.99,
                 tau=0.95,
                 entropy_coef=0.01):

        observation_space, action_space = get_env_info(env_name)
        self.num_steps = num_steps
        self.num_updates = num_updates
        self.log_frequency = log_frequency
        self.use_gae = use_gae
        self.gamma = gamma
        self.tau = tau
        self.entropy_coef = entropy_coef
        self.max_grad_norm = 0.5

        self.simulator = RolloutCollector(env_name, num_workers)
        self.eval_env = gym.make(env_name)
        self.obs_dim, self.action_dim = observation_space.shape[0], action_space.shape[0]
        self.storage = RolloutStorage(num_steps, num_workers, observation_space.shape, action_space)
        self.policy = Actor(self.obs_dim, self.action_dim)
        self.V = Critic(self.obs_dim)

        self.actor_optimizer = optim.Adam(self.policy.parameters(), lr=5e-4)
        self.critic_optimizer = optim.Adam(self.V.parameters(), lr=5e-4)

        # track statistics
        self.episode_count = 0

    def get_actions(self, obs_n):
        with torch.no_grad():
            obs_batch = torch.FloatTensor(np.stack(obs_n))
            dist = self.policy(obs_batch)
            action_sample = dist.sample()
            values = self.V(obs_batch)
            action_n = [action_sample[i].numpy() for i in range(len(action_sample))]
        return action_n, action_sample, values

    def update_storage(self, obs, actions, rewards, values, dones):
        self.episode_count += torch.sum(dones).item()
        masks = 1 - dones
        self.storage.insert(obs, actions, values, rewards, masks)

    def set_initial_observations(self, observations):
        self.storage.obs[0].copy_(observations)

    def compute_advantages(self):
        advantages = self.storage.returns[:-1] - self.storage.values[:-1]
        # standardize the advantages
        advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-5)
        return advantages

    def update(self):
        with torch.no_grad():
            next_value = self.V(self.storage.obs[-1])

        self.storage.compute_returns(next_value, self.use_gae, self.gamma, self.tau)
        self.storage.returns.mul_(0.1)
        advantages = self.compute_advantages()
        obs_batch, actions_batch, values_batch, return_batch, adv_targ = self.storage.build_batch(
            advantages)

        # Update the policy
        self.actor_optimizer.zero_grad()
        action_dist = self.policy(obs_batch)
        action_log_probs = action_dist.log_prob(actions_batch)
        objective = torch.mean(adv_targ * action_log_probs)
        policy_loss = -objective

        # compute the value loss
        self.critic_optimizer.zero_grad()
        value_loss = F.mse_loss(self.V(obs_batch), return_batch)

        # compute other losses
        entropy_loss = - torch.mean(action_dist.entropy())

        # sum the losses, backprop, and step
        net_loss = policy_loss + value_loss + self.entropy_coef * entropy_loss
        net_loss.backward()

        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.V.parameters(), self.max_grad_norm)

        self.critic_optimizer.step()
        self.actor_optimizer.step()
        return value_loss.detach().item(), -policy_loss.detach().item(), -entropy_loss.detach().item()

    def evaluate(self, n=20, render=False):
        env = self.eval_env
        action_bounds = [env.action_space.low, env.action_space.high]
        all_rewards = []
        for i in range(n):
            episode_rewards = []
            state = env.reset()
            terminal = False
            while not terminal:
                dist = self.policy(torch.FloatTensor(state).view(1, -1))
                action = dist.sample().numpy().reshape(-1)
                action = np.clip(action, action_bounds[0], action_bounds[1])
                next_state, reward, terminal, info = env.step(action)
                episode_rewards.append(reward)
                state = next_state
                if render:
                    fps = 8.0
                    env.render()
                    time.sleep(1 / fps)
            all_rewards.append(np.sum(episode_rewards))
        all_rewards = np.array(all_rewards)
        env.reset()
        return all_rewards

    def __iter__(self):
        obs_n = self.simulator.reset()
        for u in range(self.num_updates):
            self.set_initial_observations(torch.FloatTensor(np.stack(obs_n)))
            for t in range(self.num_steps):
                # Compute actions using policy given latest observation
                action_n, actions, values = self.get_actions(obs_n)

                # Give action to each worker and take an environment step
                obs_n, reward_n, done_n = self.simulator.step(action_n)

                observations = torch.FloatTensor(np.stack(obs_n))
                rewards = torch.FloatTensor(np.vstack(reward_n))
                dones = torch.FloatTensor(np.vstack(done_n))

                # Update the storage
                self.update_storage(observations, actions, rewards, values, dones)

            value_loss, objective, mean_policy_entropy = self.update()
            self.storage.after_update()

            if (u + 1) % self.log_frequency == 0:
                eval_episode_returns = self.evaluate()
                yield self.episode_count, eval_episode_returns, value_loss, objective, mean_policy_entropy
