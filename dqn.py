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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('load', type=bool, help='load file or not', default=False)
args = parser.parse_args()
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

board_input = Input(shape=env.observation_space.shape)
last_action = Input(shape=(1, ))

x = Conv2D(filters=128, kernel_size=(2, 2), activation="tanh",
           data_format="channels_first")(board_input)
x = Conv2D(filters=64, kernel_size=(2, 2),
                 activation="tanh", data_format="channels_first")(x)
x = Conv2D(filters=64, kernel_size=(2, 2),
           activation="tanh", data_format="channels_first")(x)
x = Flatten()(x)
x = Dense(512, activation="tanh")(x)
x = Dense(64, activation="tanh")(x)
# x = Concatenate()([x, last_action])
x = Dense(64, activation="tanh")(x)
x = Dense(64, activation="tanh")(x)
x = Dense(nb_actions, activation="linear")(x)
model = Model(inputs=board_input, outputs=x)
model.summary()

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
def compute_q_values(self, state):
    q_values = self.compute_batch_q_values(state).flatten()
    assert q_values.shape == (self.nb_actions,)
    return q_values


def backward(self, reward, terminal):
    # Store most recent experience in memory.
    if self.step % self.memory_interval == 0:
        self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                           training=self.training)

    metrics = [np.nan for _ in self.metrics_names]
    if not self.training:
        # We're done here. No need to update the experience memory since we only use the working
        # memory to obtain the state over the most recent observations.
        return metrics

    # Train the network on a single stochastic batch.
    if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
        experiences = self.memory.sample(self.batch_size)
        assert len(experiences) == self.batch_size

        # Start by extracting the necessary parameters (we use a vectorized implementation).
        state0_batch = []
        reward_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        for e in experiences:
            state0_batch.append(e.state0)
            state1_batch.append(e.state1)
            reward_batch.append(e.reward)
            action_batch.append(e.action)
            terminal1_batch.append(0. if e.terminal1 else 1.)

        # Prepare and validate parameters.
        state0_batch = self.process_state_batch(state0_batch)
        state0_batch = np.reshape(state0_batch, (self.batch_size, 2, 12, 12))
        state1_batch = self.process_state_batch(state1_batch)
        state1_batch = np.reshape(state1_batch, (self.batch_size, 2, 12, 12))
        terminal1_batch = np.array(terminal1_batch)
        reward_batch = np.array(reward_batch)
        assert reward_batch.shape == (self.batch_size,)
        assert terminal1_batch.shape == reward_batch.shape
        assert len(action_batch) == len(reward_batch)

        # Compute Q values for mini-batch update.
        if self.enable_double_dqn:
            # According to the paper "Deep Reinforcement Learning with Double Q-learning"
            # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
            # while the target network is used to estimate the Q value.
            q_values = self.model.predict_on_batch(state1_batch)
            assert q_values.shape == (self.batch_size, self.nb_actions)
            actions = np.argmax(q_values, axis=1)
            assert actions.shape == (self.batch_size,)

            # Now, estimate Q values using the target network but select the values with the
            # highest Q value wrt to the online model (as computed above).
            target_q_values = self.target_model.predict_on_batch(
                state1_batch)
            assert target_q_values.shape == (
                self.batch_size, self.nb_actions)
            q_batch = target_q_values[range(self.batch_size), actions]
        else:
            # Compute the q_values given state1, and extract the maximum for each sample in the batch.
            # We perform this prediction on the target_model instead of the model for reasons
            # outlined in Mnih (2015). In short: it makes the algorithm more stable.
            target_q_values = self.target_model.predict_on_batch(
                state1_batch)
            assert target_q_values.shape == (
                self.batch_size, self.nb_actions)
            q_batch = np.max(target_q_values, axis=1).flatten()
        assert q_batch.shape == (self.batch_size,)

        targets = np.zeros((self.batch_size, self.nb_actions))
        dummy_targets = np.zeros((self.batch_size,))
        masks = np.zeros((self.batch_size, self.nb_actions))

        # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
        # but only for the affected output units (as given by action_batch).
        discounted_reward_batch = self.gamma * q_batch
        # Set discounted reward to zero for all states that were terminal.
        discounted_reward_batch *= terminal1_batch
        assert discounted_reward_batch.shape == reward_batch.shape
        Rs = reward_batch + discounted_reward_batch
        for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
            # update action with estimated accumulated reward
            target[action] = R
            dummy_targets[idx] = R
            mask[action] = 1.  # enable loss for this specific action
        targets = np.array(targets).astype('float32')
        masks = np.array(masks).astype('float32')

        # Finally, perform a single update on the entire batch. We use a dummy target since
        # the actual loss is computed in a Lambda layer that needs more complex input. However,
        # it is still useful to know the actual target to compute metrics properly.
        ins = [state0_batch] if type(
            self.model.input) is not list else state0_batch
        metrics = self.trainable_model.train_on_batch(
            ins + [targets, masks], [dummy_targets, targets])
        metrics = [metric for idx, metric in enumerate(
            metrics) if idx not in (1, 2)]  # throw away individual losses
        metrics += self.policy.metrics
        if self.processor is not None:
            metrics += self.processor.metrics

    if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
        self.update_target_model_hard()

    return metrics

DQNAgent.compute_q_values = compute_q_values
DQNAgent.backward = backward

memory=SequentialMemory(limit = 50000, window_length = 1)
policy=MyPolicy(env)
dqn=DQNAgent(model = model, nb_actions = nb_actions, memory = memory, nb_steps_warmup = 10000,
            target_model_update = 1e-3, policy = policy, enable_dueling_network = False)
dqn.compile(Adam(lr=1e-4), metrics = ['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

if args.load:
    dqn.load_weights('dqn_{}_weights.h5f'.format(ENV_NAME))
dqn.fit(env, nb_steps = 250000, visualize = False, verbose = 2)
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite = True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes = 5, visualize = False)
