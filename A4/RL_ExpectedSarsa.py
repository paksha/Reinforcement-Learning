import numpy as np
import pandas as pd


class ExpectedSarsa:
    # lr = 0.01, decay = 0.9, epsilon = 0.1 --> didn't converge fully in 2k episodes
    # learning_rate=0.05, reward_decay=0.95, e_greedy=0.1 --> still spiking at the end
    # increasing learning_rate to 0.5 helped a lot in learning faster I think
    def __init__(self, actions, learning_rate=1, reward_decay=0.9, e_greedy=0.1):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.display_name="Expected SARSA"

    '''Choose the next action to take given the observed state using an epsilon greedy policy'''
    def choose_action(self, observation):
        self.check_state_exist(observation)
 
        if np.random.uniform() >= self.epsilon:          
            state_action = self.q_table.loc[observation, :]
            # Choose action with highest reward from list of actions corresponding to state
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # Choose random action. This is the epsilon case.
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        action_values = self.q_table.loc[s_,:]
        total_value, max_val = sum(action_values), max(action_values)
        num_max = len(action_values[action_values == max_val])
        total_max_value = max_val * num_max
        exp_q = (((1-self.epsilon)/num_max) + (self.epsilon/len(action_values))) * total_max_value
        exp_q += (self.epsilon/len(self.actions)) * (total_value - total_max_value)
        a_ = self.choose_action(str(s_))
        q_initial = self.q_table.loc[s, a]
        self.q_table.loc[s, a] = q_initial + (self.lr * (r + (self.gamma * exp_q) - q_initial))
        return s_, a_

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
    
    def eps_decay(self):
        self.epsilon *= 0.99
