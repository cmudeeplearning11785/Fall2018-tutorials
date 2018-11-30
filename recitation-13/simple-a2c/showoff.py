import gym
import torch
import time
from a2c.model import Actor
import argparse
import numpy as np

class Showoff:

    def __init__(self, env_name, checkpoint_path, record=False):
        self.env = gym.make(env_name)
        if record:
            self.env = gym.wrappers.Monitor(self.env, "recording")
        state_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        self.policy = Actor(state_dim, act_dim)

        self.policy.load_state_dict(torch.load(checkpoint_path))
        print(self.policy.policy_dist.logstd.data)

        self.action_bounds = [self.env.action_space.low, self.env.action_space.high]


    def showoff(self, n):
        for t in range(n):
            states, actions, rewards = self.rollout()
            print("\n({})\n"
                  "\tReturn: {}".format(t, rewards.sum().item()))

    def rollout(self):
        terminal = False
        rewards = []
        states = []
        actions = []
        state = self.env.reset()
        while not terminal:
            self.env.render()
            with torch.no_grad():
                action = self.policy(torch.FloatTensor(state.reshape(1, -1))).sample()
            action = action.numpy().reshape(-1)
            action = np.clip(action, self.action_bounds[0], self.action_bounds[1])
            next_state, reward, terminal, _ = self.env.step(action)
            state = next_state
            states.append(state)
            rewards.append(reward)
            actions.append(action)
            time.sleep(0.05)
            if terminal:
                break
        return torch.tensor(states, dtype=torch.float32), \
               torch.tensor(actions, dtype=torch.int64).unsqueeze(1), \
               torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    parser.add_argument("--env", type=str, default="LunarLanderContinuous-v2")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--record", action="store_true", default=False)

    args = parser.parse_args()
    Showoff(args.env, args.file, record=args.record).showoff(args.n)