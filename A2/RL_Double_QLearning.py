import numpy as np
import pandas as pd


class DoubleQLearning:
    
    def __init__(self, actions, learning_rate=0.05, reward_decay=0.95, e_greedy=0.1):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.q_ = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.display_name="Double Q-Learning"

    '''Choose the next action to take given the observed state using an epsilon greedy policy'''
    def choose_action(self, observation):
        self.check_state_exist(observation)
        vals = self.q.loc[observation, :] + self.q_.loc[observation, :]
        if np.random.uniform() >= self.epsilon:
            action = np.random.choice(vals[vals == np.max(vals)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        a_ = self.choose_action(str(s_))
        if np.random.uniform() < 0.5:
            self.update_q(self.q, self.q_, s, a, r, s_)
        else:
            self.update_q(self.q_, self.q, s, a, r, s_)
        return s_, a_
    
    def update_q(self, q, q_, s, a, r, s_):
        q_initial = q.loc[s, a]
        state_action = q.loc[s_,:]
        a_q = np.random.choice(state_action[state_action == np.max(state_action)].index)
        q.loc[s, a] = q_initial + self.lr * (r + (self.gamma * q_.loc[s_, a_q]) - q_initial)

    def check_state_exist(self, state):
        if state not in self.q.index:
            self.q = self.q.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q.columns,
                    name=state,
                )
            )
        if state not in self.q_.index:
            self.q_ = self.q_.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_.columns,
                    name=state
                )
            )

    def eps_decay(self):
        self.epsilon *= 0.99