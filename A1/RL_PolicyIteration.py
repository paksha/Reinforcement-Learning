import numpy as np
import pandas as pd
from copy import deepcopy

class AsyncPolicyIteration:

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        # Action-Value Function
        self.q_table = {}
        # V(s) --> R number
        self.V = {}
        self.theta = 0
        self.display_name="Asynchronous Policy Iteration"

    '''Choose the next action to take given the observed state using an epsilon greedy policy'''
    def choose_action(self, observation):
        self.check_state_exist(observation)
 
        # (FIXED) BUG: Epsilon should be .1 and signify the small probability of NOT choosing max action
        if np.random.uniform() >= self.epsilon:
            action = self.q_table[observation]
        else:
            # Choose random action. This is the epsilon case.
            action = np.random.choice(self.actions)
        return action

    def policyEval(self, ):
        states = list(self.q_table.keys())
        V = np.zeros(len(states)) # Start with arbitrary value function
        while (True):
            delta = 0
            for state in states:
                v = V[state]
                action = self.q_table[state]

                

    '''Update the Q(S,A) state-action value table using the latest experience
       This is a not a very good learning update 
    '''
    def learn(self, s, a, r, s_, env):
        self.check_state_exist(s_)
        if s_ != 'terminal':
            a_ = self.choose_action(str(s_))
            self.V[s] = r + self.gamma * self.V[s_]
        else:
            self.V[s] = r  # next state is terminal
        A = self.lookAhead(env, s)
        self.q_table[s] = np.argmax(A)
        return s_, a_


    '''States are dynamically added to the Q(S,A) table as they are encountered'''
    def check_state_exist(self, state):
        if state not in self.q_table:
            # append new state to q table
            self.q_table[state] = np.random.choice(self.actions)
        if state not in self.V:
            self.V[state] = 0

    '''step - definition of one-step dynamics function'''
    def step(self, env, state, action):
        temp_s = state.strip('][').split(', ')
        s = [float(coord) for coord in temp_s]
        base_action = np.array([0, 0])
        UNIT, MAZE_H, MAZE_W = env.UNIT, env.MAZE_H, env.MAZE_W
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        s_ =  self.move(s, base_action)

        # call the reward function
        reward, done, _ = env.computeReward(s, action, s_)
        return str(s_), reward, done

    def lookAhead(self, env, s):
        A = np.zeros(len(self.actions))
        for a in range(len(self.actions)):
            s_, reward, _ = self.step(env=env, state=s, action=a)
            self.check_state_exist(state=s_)
            A[a] = reward + self.gamma * self.V[s_]
        return A

    def move(self, s, action):
        s[0] += action[0] # update x0
        s[1] += action[1] # update y0
        s[2] += action[0] # update x1
        s[3] += action[1] # update y1
        return s