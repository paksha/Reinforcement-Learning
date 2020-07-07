import numpy as np
from utility import step

class AsyncPolicyIteration:

    def __init__(self, env, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1):
        self.actions = list(range(env.n_actions)) 
        self.env = env
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.pi = {}
        # V(s) --> R number
        self.V = {}
        self.theta = 0
        self.display_name="Asynchronous Policy Iteration"

    '''Choose the next action to take given the observed state using an epsilon greedy policy'''
    def choose_action(self, observation):
        self.check_state_exist(observation)
 
        if np.random.uniform() >= self.epsilon:
            action = self.pi[observation]
        else:
            # Choose random action. This is the epsilon case.
            action = np.random.choice(self.actions)
        return action


    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        if s_ != 'terminal':
            a_ = self.choose_action(str(s_))
            self.V[s] = r + self.gamma * self.V[s_]
        else:
            self.V[s] = r  # next state is terminal
        A = self.lookAhead(s)
        self.pi[s] = np.argmax(A)
        return s_, a_


    '''States are dynamically added to the tables as they are encountered'''
    def check_state_exist(self, state):
        if state not in self.pi:
            # append new state to q table
            self.pi[state] = np.random.choice(self.actions)
        if state not in self.V:
            self.V[state] = 0

    def lookAhead(self, s):
        A = np.zeros(len(self.actions))
        for a in range(len(self.actions)):
            s_, reward, _ = step(env=self.env, state=s, action=a)
            self.check_state_exist(state=s_)
            A[a] = reward + self.gamma * self.V[s_]
        return A