import numpy as np
import pandas as pd

class rlalgorithm:

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
        self.display_name="Policy Iteration"

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
        reverse_a = self.reverseAction(a)
        sx, rx, dx = self.step(env, reverse_a, moving_back=True) # Puts the agent back to s instead of s_
        A = self.lookAhead(env)
        self.q_table[s] = np.argmax(A)
        sx, rx, dx = self.step(env, a, moving_back=True) # Put agent back to s_ from s
        return s_, a_


    '''States are dynamically added to the Q(S,A) table as they are encountered'''
    def check_state_exist(self, state):
        if state not in self.q_table:
            # append new state to q table
            self.q_table[state] = np.random.choice(self.actions)
        if state not in self.V:
            self.V[state] = 0

    '''step - definition of one-step dynamics function'''
    def step(self, env, action, moving_back=False):
        s = env.canvas.coords(env.agent)
        base_action = np.array([0, 0])
        UNIT = env.UNIT
        MAZE_H = env.MAZE_H
        MAZE_W = env.MAZE_W
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

        env.canvas.move(env.agent, base_action[0], base_action[1])  # move agent

        s_ = env.canvas.coords(env.agent)  # next state

        # call the reward function
        reward, done, reverse = env.computeReward(s, action, s_)
        if (reverse):
            env.canvas.move(env.agent, -base_action[0], -base_action[1])  # move agent back
            s_ = env.canvas.coords(env.agent)
        elif not moving_back:
            env.canvas.move(env.agent, -base_action[0], -base_action[1])

        return str(s_), reward, done

    def lookAhead(self, env):
        A = np.zeros(len(self.actions))
        for a in range(len(self.actions)):
            s_, reward, _ = self.step(env, a)
            self.check_state_exist(s_)
            A[a] = reward + self.gamma * self.V[s_]
        return A

    def reverseAction(self, a):
        if a == 0:
            return 1
        elif a == 1:
            return 0
        elif a == 2:
            return 3
        elif a == 3:
            return 2