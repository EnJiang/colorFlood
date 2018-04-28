import numpy as np
import gym

from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.processors import MultiInputProcessor

from env import Env
from greedy import greedy

import random

DEBUG = False


class MyPolicy(EpsGreedyQPolicy):
    def __init__(self, env, eps=.5):
        super().__init__()
        self.eps = eps
        self.env = env
        self.step = 0
        self.decay_step = 1000

    def select_action(self, q_values):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        rand = random.random()
        if 0 < rand < 0.05:  # 5% rand
            action = np.random.random_integers(0, nb_actions - 1)
            if DEBUG:
                print("rand")
        elif 0.05 < rand < 0.15:  # 10% greedy
            if DEBUG:
                print("geedy")
            try:
                action = greedy(self.env.game, 1)[0] - 1
            except:
                action = np.random.random_integers(0, nb_actions - 1)
        else:
            if DEBUG:
                print("net")
            action = np.argmax(q_values)

        return action

ENV_NAME = 'colorflood'

# Get the environment and extract the number of actions.
env = Env()
nb_actions = 6
# Next, we build a very simple model.

board_input = Input(env.observation_space.shape)
last_action = Input(shape=(1, ))

x = Conv2D(filters=64, kernel_size=(2, 2), activation="relu",
           data_format="channels_first")(board_input)
x = Conv2D(filters=32, kernel_size=(2, 2),
                 activation="relu", data_format="channels_first")(x)
x = Conv2D(filters=32, kernel_size=(2, 2),
                 activation="relu", data_format="channels_first")(x)
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dense(64, activation="relu")(x)
# x = Concatenate()([x, last_action])
x = Dense(64, activation="relu")(x)
x = Dense(64, activation="relu")(x)
x = Dense(nb_actions, activation="linear")(x)
model = Model(inputs=board_input, outputs=x)
model.summary()

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory=SequentialMemory(limit = 50000, window_length = 1)
policy=MyPolicy(env)
dqn=DQNAgent(model = model, nb_actions = nb_actions, memory = memory, nb_steps_warmup = 10000,
            target_model_update = 1e-3, policy = policy, enable_dueling_network = False)
dqn.compile(Adam(lr=1e-4), metrics = ['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

# dqn.load_weights('dqn_{}_weights.h5f'.format(ENV_NAME))
dqn.fit(env, nb_steps = 250000, visualize = False, verbose = 2)
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite = True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes = 5, visualize = False)
