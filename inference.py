import argparse

import numpy as np
from stable_baselines3.common.env_util import make_vec_env
import utils.import_envs  # noqa: F401 pytype: disable=import-error
from utils.utils import ALGOS


def evaluate(env, model, num_steps=1000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """
    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

        # Stats
        episode_rewards[-1] += reward
        if done:
            # DO NOT NEED env.reset() HERE!
            episode_rewards.append(0.0)
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = round(float(np.mean(episode_rewards[-100:])), 3)
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

    return mean_100ep_reward


def get_prediction():
    obs = {
            "grid": (-2, 0, -2, -1, 0, 0, -2, 1, 1, -1, -1, 1, 0, -2, 0, -1, -1, -1, 1, 1, -1, 0, -1, -1, -2),
            "positions": ((10, 0), (0, 0)),
            "modificators": (0, 0, 0, 0, 0),
            "my_mods": (0, 0, 0, 0),
            "opp_mods": (0, 0, 0, 0),
        }
    action, _states = model.predict(obs, deterministic=True)
    print("action=", action)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False,
                        choices=list(ALGOS.keys()))
    parser.add_argument("--env", type=str, default="JavaBot-v1", help="environment ID")
    parser.add_argument("--model-path", help="Path to model", type=str, default="rl_model_24_2M_steps")
    parser.add_argument("--eval-episodes", help="Number of episodes to use for evaluation", default=5, type=int)
    args = parser.parse_args()

    env_id = args.env
    vec_env = make_vec_env(env_id, n_envs=1)
    model = ALGOS[args.algo].load(args.model_path, device='cpu')
    # evaluate(vec_env, model)
    get_prediction()
