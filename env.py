import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from game import Game

class Env(gym.Env):
    def __init__(self):
        self.game = Game()

    @property
    def action_space(self):
        return np.array([0, 1, 2, 3, 4, 5])

    @property
    def observation_space(self):
        return self.obs(self.game.baseColor)

    def obs(self, color):
        features = []

        last_action = (np.zeros((1, 12, 12)) + color - 2.5) / 2.5
        features.append(last_action)

        mainBorad = (np.reshape(self.game.mainBorad, (1, 12, 12)) - 3.5) / 3.5
        features.append(mainBorad)

        diff = self.game.mainBorad - color
        diff = np.reshape(diff, (1, 12, 12)) / 10
        features.append(diff)

        target_board = np.reshape(self.game.targetBoard, (1, 12, 12))
        features.append(target_board)

        ob = np.concatenate(features, axis=0)
        return ob

    def step(self, action):
        game = self.game

        game.change(action + 1)

        next_state = game.mainBorad
        done = game.isOver()

        if done and game.targetArea() == 144: # really done
            reward = 1000 - game.step * 5
        if done and game.targetArea() < 144:  # step overflow
            reward = game.targetArea()
        if not done:
            reward = 0

        ob = self.obs(action)

        return ob, reward, done, {}

    def reset(self):
        self.game = Game()
        return self.observation_space

    # render environment
    def render(self, **kwargs):
        pass
