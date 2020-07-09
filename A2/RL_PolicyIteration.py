import numpy as np
from utility import step

class AsyncPolicyIteration:

    def __init__(self, env, learning_rate=0.01, reward_decay=0.95, e_greedy=0.1):
        self.actions = list(range(env.n_actions)) 
        self.env = env # This would be the maze environment
        self.lr = learning_rate # Not used
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.pi = {} # Pi(s) --> Action
        self.V = {} # V(s) --> Real number
        self.train = True # This is used to alternate between policy improvement and evaluation in the Learn function
        self.pstable = False # This represents the policy being stable, and so no need to do more work on it
        # This is a list of states which I will not update the value function for after initializing them to 0
        self.block = [str(self.env.canvas.coords(w)) for w in self.env.wallblocks] + [str(self.env.canvas.coords(w)) for w in self.env.pitblocks] + [str(self.env.canvas.coords(self.env.goal))]
        self.display_name="Asynchronous Policy Iteration"

    '''Choose the next action to take given the observed state using an epsilon greedy policy'''
    def choose_action(self, observation):
        self.check_state_exist(observation)
        # If the policy is stable, do not go to the epsilon case
        if np.random.uniform() >= self.epsilon or self.pstable:
            action = self.pi[observation]
        else: # Choose random action. This is the epsilon case.
            action = np.random.choice(self.actions)
        return action


    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        if (self.train and not self.pstable): # If the policy is NOT stable and it's time to improve the policy
            self.updatePolicy()
            self.train = False
        elif not self.pstable: # If policy is NOT stable and it's time to evaluate the policy
            self.updateValues()
            self.train = True
        a_ = self.choose_action(s_)
        return s_, a_

    def updateValues(self):
        for s in list(self.V.keys()):
            if s not in self.block: # If s is a normal state that the agent can be in (non-terminal and not blocking)
                action = self.pi[s]
                s_, reward, _ = step(env=self.env, state=s, action=action)
                self.check_state_exist(s_)
                self.V[s] = reward + self.gamma * self.V[s_] # Update using Bellman equation

    def updatePolicy(self):
        stable = True
        states = list(self.pi.keys())
        for s in states:
            if s not in self.block:
                old = self.pi[s]
                A = self.lookAhead(s)
                self.pi[s] = np.argmax(A) # Update the policy to return the best action for state s
                if (old != self.pi[s]): # If the best action is not the same as the original policy, then it's unstable
                    stable = False
        if (len(states) == self.env.MAZE_H * self.env.MAZE_W):
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
            '''
            I am using my own step function here. Note that it doesn't reverse back if we run into a wall,
            but since I do not update the values for terminal or blocking states, the Bellman update
            still works correctly as self.V[s_] is 0 for those states.
            '''
            s_, reward, _ = step(env=self.env, state=s, action=a)
            self.check_state_exist(state=s_)
            A[a] = reward + self.gamma * self.V[s_]
        return A