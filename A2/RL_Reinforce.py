import numpy as np
import pandas as pd
import copy
import pylab
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K

class Reinforce:
    
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0):
        self.state_size = 100
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.model = self.build_model()
        self.opt = self.optimizer()
        self.states = []
        self.actions = []
        self.rewards = []
        self.display_name="Reinforce"

    def choose_action(self, observation):
        policy = self.model.predict(observation)[0]
        return np.random.choice(len(self.actions), 1, p=policy)[0]

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(len(self.actions), activation='softmax'))
        model.summary()
        return model

    def optimizer(self):
        action = K.placeholder(shape=[None, len(self.actions)])
        discounted_rewards = K.placeholder(shape=[None, ])

        # Calculate cross entropy error function
        action_prob = K.sum(action * self.model.output, axis=1)
        cross_entropy = K.log(action_prob) * discounted_rewards
        loss = -K.sum(cross_entropy)

        # create training function
        optimizer = Adam(lr=self.lr)
        updates = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        train = K.function([self.model.input, action, discounted_rewards], [], updates=updates)

        return train

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards
    
    def train_model(self):
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        self.opt([self.states, self.actions, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []

    def eps_decay(self):
        self.train_model()