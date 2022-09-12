import random

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

from bot.javaconnector import JavaConnector

MAX_STEPS = 150*60*3
OBSERVABLE_SIZE = 2


class JavaBot(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}

    # Define constants for clearer code

    def __init__(self, grid_size=100):
        self.seed_internal = None
        super(JavaBot, self).__init__()
        # Size of the 1D0-grid
        self.grid_size = grid_size

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # The observation will be the coordinate of the agent
        self.observation_space = spaces.Dict(
            {
             "grid": spaces.Box(low=-2, high=2, shape=((OBSERVABLE_SIZE*2+1) * (OBSERVABLE_SIZE*2+1), ), dtype=np.int8),
             # "positions": spaces.Box(low=0, high=100, shape=(2, 2), dtype=int),
             # "positions": spaces.Box(low=0, high=9, shape=(2, 2), dtype=int),
             "positions": spaces.Box(low=0, high=100, shape=(2, 2), dtype=int),
             "modificators": spaces.MultiBinary(5),
             # "mod_positions": spaces.Box(low=0, high=100, shape=(5, 2)),
             "my_mods": spaces.MultiBinary(4),
             "opp_mods": spaces.MultiBinary(4),
             }
        )
        self.steps = 0

    def reset(self):
        self.java_env.reset_season()
        obs = self.java_env.get_observation(OBSERVABLE_SIZE)
        return obs

    def seed(self, seed=None):
        self.seed_internal = seed
        self.java_env = JavaConnector(port=25331+seed*2)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.steps += 1
        java_env = self.java_env
        # action = (random.randint(-1, 1), random.randint(-1, 1))
        java_env.make_one_step(action)
        # Account for the boundaries of the grid
        obs = java_env.get_observation(OBSERVABLE_SIZE)
        done = java_env.is_round_over()
        # done = False

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = java_env.get_reward()
        my_x, my_y = obs["positions"][0]
        # if (my_x <= 1 and action[0] < 0) or (my_x >= 9 and action[0] > 0) or \
        #         (my_y <= 1 and action[1] < 0) or (my_y >= 9 and action[1] > 0):
        #     reward -= 1
        #     print("the Wall!")
        # Optionally we can pass additional info, we are not using that for now
        info = {}
        if self.steps == MAX_STEPS:
            self.steps = 0
            done = True
        # reward -= 0.01
        return obs, reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        print("." * self.agent_pos, end="")
        print("x", end="")
        print("." * (self.grid_size - self.agent_pos))

    def close(self):
        pass


def check():
    from stable_baselines3.common.env_checker import check_env
    env = JavaBot()
    # If the environment don't follow the interface, an error will be thrown
    check_env(env, warn=True)


if __name__ == "__main__":
    check()
