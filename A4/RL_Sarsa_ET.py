import numpy as np
import pandas as pd


class SarsaET:

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1, ld=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.ld = ld
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.e_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.display_name="SARSA Eligibility Traces"

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() >= self.epsilon:          
            state_action = self.q_table.loc[observation, :]
            return np.random.choice(state_action[state_action == np.max(state_action)].index)
        return np.random.choice(self.actions)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        a_ = self.choose_action(str(s_))
        q_initial = self.q_table.loc[s, a]
        q_target = r + (self.gamma * self.q_table.loc[s_, a_])
        delta = q_target - q_initial
        self.e_table.loc[s, a] += 1
        self.q_table += self.lr * delta * self.e_table
        self.e_table = self.gamma * self.ld * self.e_table
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
        if state not in self.e_table.index:
            self.e_table = self.e_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.e_table.columns,
                    name=state,
                )
            )

    def eps_decay(self):
        self.epsilon *= 0.99