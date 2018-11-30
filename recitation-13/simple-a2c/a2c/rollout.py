import numpy as np
import ray
import gym

@ray.remote(num_cpus=1)
class Worker:

    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.obs = None
        self.reset()
        self.action_bounds = [self.env.action_space.low, self.env.action_space.high]

    def step(self, action):
        action = np.clip(action, self.action_bounds[0], self.action_bounds[1])
        next_obs, reward, done, _ = self.env.step(action)
        self.obs = next_obs
        if done:
            self.reset()
        return [self.obs, reward, float(done)]

    def reset(self):
        self.obs = self.env.reset()
        return self.obs



class RolloutCollector:

    def __init__(self, env_name, num_workers):
        self.collectors = [Worker.remote(env_name) for _ in range(num_workers)]

    def step(self, actions):
        """ actions for each worker """
        step_data = ray.get([collector.step.remote(action) for collector, action in zip(self.collectors, actions)])
        obs_n, reward_n, done_n = zip(*step_data)
        return obs_n, reward_n, done_n

    def reset(self):
        """ reset all workers and return their first obs """
        obs_n = ray.get([collector.reset.remote() for collector in self.collectors])
        return obs_n


