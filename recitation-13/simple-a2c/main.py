import ray
from a2c.train import A2C
import torch
import numpy as np





if __name__ == "__main__":
    ray.init()
    env_name = "LunarLanderContinuous-v2"
    trainer = A2C(env_name=env_name,
                  num_steps=50,
                  num_workers=5,
                  num_updates=5000,
                  log_frequency=20,
                  use_gae=True,
                  gamma=0.999,
                  tau=0.95,
                  entropy_coef=0.01)

    best_eval_mean_return = -np.inf
    for t, log_data in enumerate(trainer):
        episode_count, eval_episode_returns, value_loss, objective, mean_policy_entropy = log_data

        print("({})\n"
              "\tEpisode Count: {}\n"
              "\tMean Episode Return: {}\n"
              "\tMax/Min Episode Return: {} / {}\n"
              "\tStd Episode Return: {}\n"
              "\tValue Loss: {}\n"
              "\tJ(theta): {}"
              "\tEntropy: {}".format(t, episode_count, eval_episode_returns.mean(),eval_episode_returns.max(),
                                     eval_episode_returns.min(), np.std(eval_episode_returns), value_loss, objective, mean_policy_entropy))

        if eval_episode_returns.mean() > best_eval_mean_return:
            best_eval_mean_return = eval_episode_returns.mean()
            torch.save(trainer.policy.state_dict(), "best_policy7.pt")
            print("Saved policy parameters")