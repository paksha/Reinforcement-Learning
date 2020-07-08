import numpy as np
from utility import step

class AsyncPolicyIteration:

    def __init__(self, env, learning_rate=0.01, reward_decay=0.99, e_greedy=0.05):
        self.actions = list(range(env.n_actions)) 
        self.env = env
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.pi = {}
        # V(s) --> R number
        self.V = {}
        self.train = True
        self.pstable = False
        self.display_name="Asynchronous Policy Iteration"

    '''Choose the next action to take given the observed state using an epsilon greedy policy'''
    def choose_action(self, observation):
        self.check_state_exist(observation)
 
        if np.random.uniform() >= self.epsilon or self.pstable:
            action = self.pi[observation]
        else:
            # Choose random action. This is the epsilon case.
            action = np.random.choice(self.actions)
        return action


    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        if (self.train and not self.pstable):
            self.updatePolicy()
            self.train = False
        elif not self.pstable:
            self.updateValues()
            self.train = True
        a_ = self.choose_action(s_)
        return s_, a_

    def updateValues(self):
        block = [str(self.env.canvas.coords(w)) for w in self.env.wallblocks] + [str(self.env.canvas.coords(w)) for w in self.env.pitblocks] + [str(self.env.canvas.coords(self.env.goal))]
        for s in list(self.V.keys()):
            if s not in block:
                action = self.pi[s]
                s_, reward, _ = step(env=self.env, state=s, action=action)
                self.check_state_exist(s_)
                self.V[s] = reward + self.gamma * self.V[s_]

    def updatePolicy(self):
        block = [str(self.env.canvas.coords(w)) for w in self.env.wallblocks] + [str(self.env.canvas.coords(w)) for w in self.env.pitblocks] + [str(self.env.canvas.coords(self.env.goal))]
        stable = True
        states = list(self.pi.keys())
        for s in states:
            if s not in block:
                old = self.pi[s]
                A = self.lookAhead(s)
                self.pi[s] = np.argmax(A)
                if (old != self.pi[s]):
                    stable = False
        if (len(states) == 100):
            self.pstable = stable

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