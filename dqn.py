import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

from env import Env
from greedy import greedy

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

        rand = np.random.uniform()
        if 0 < rand < 0.05: # 5% rand
            action = np.random.random_integers(0, nb_actions - 1)
        elif 0.05 < rand < 0.15: # 10% greedy
            try:
                action = greedy(self.env.game, 1)[0] - 1
            except:
                action = np.random.random_integers(0, nb_actions - 1)
        else:
            action = np.argmax(q_values)

        return action



ENV_NAME = 'colorflood'

# Get the environment and extract the number of actions.
env = Env()
nb_actions = 6
# Next, we build a very simple model.

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2, 2), activation="relu", input_shape=(1, ) + env.observation_space.shape,
                 data_format="channels_first"))
# model.add(MaxPool2D(2, 2)),
# model.add(Conv2D(filters=8, kernel_size=(3, 3),
#                  activation="relu", data_format="channels_first"))
# model.add(MaxPool2D(2, 2))
model.add(Conv2D(filters=64, kernel_size=(2, 2),
                 activation="relu", data_format="channels_first"))
model.add(Conv2D(filters=64, kernel_size=(2, 2),
                 activation="relu", data_format="channels_first"))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
model.summary()


# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = MyPolicy(env)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
               target_model_update=1e-3, policy=policy, enable_dueling_network=True)
dqn.compile(Adam(lr=1e-4), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

# dqn.load_weights('dqn_{}_weights.h5f'.format(ENV_NAME))
dqn.fit(env, nb_steps=500000, visualize=False, verbose=2)
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=False)
