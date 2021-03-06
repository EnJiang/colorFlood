import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from game import Game

class Env(gym.Env):
    def __init__(self, size=12):
        self.size = size
        self.game = Game(size=size)
        self._last_obs = None

    @property
    def action_space(self):
        return np.array([0, 1, 2, 3, 4, 5])

    @property
    def observation_space(self):
        return self.obs(self.game.baseColor)

    @property
    def last_obs(self):
        return self._last_obs

    @property
    def last_action_index(self):
        return self.game.baseColor - 1

    def obs(self, color):
        size = self.size

        features = []

        last_action = (np.zeros((1, size, size)) + color - 2.5) / 2.5
        features.append(last_action)

        mainBorad = (np.reshape(self.game.mainBorad, (1, size, size)) - 3.5) / 3.5
        features.append(mainBorad)

        diff = self.game.mainBorad - color
        diff = np.reshape(diff, (1, size, size)) / 10
        features.append(diff)

        target_board = np.reshape(self.game.targetBoard, (1, size, size))
        features.append(target_board)

        ob = np.concatenate(features, axis=0)

        # book keeping
        self._last_obs = ob
        return ob

    def step(self, action):
        info = {}
        game = self.game

        game.change(action + 1)

        next_state = game.mainBorad
        done = game.isOver()

        if done and game.targetArea() == game.point_num: # really done
            reward = 1000 - game.step * 5
            info["overflow"] = False
        if done and game.targetArea() < game.point_num:  # step overflow
            reward = game.targetArea()
            info["overflow"] = True
        if not done:
            reward = 0

        ob = self.obs(action)

        return ob, reward, done, info

    def reset(self):
        self.game = Game(size=self.size)
        return self.observation_space

    # render environment
    def render(self, **kwargs):
        pass
