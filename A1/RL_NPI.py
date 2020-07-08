import numpy as np
from utility import step

class NormalPolicyIteration:
    # I set the epsilon to 0 since I want to just use the calculated optimal policy
    def __init__(self, env, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1):
        self.actions = list(range(env.n_actions)) 
        self.env = env
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.pi = {}
        # V(s) --> R number
        self.V = {}
        self.theta = 0.001
        self.train = True
        self.stable = False
        self.statesAdded = False
        self.display_name="Policy Iteration"

    '''Choose the next action to take given the observed state using an epsilon greedy policy'''
    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() >= self.epsilon:
            action = self.pi[observation]
        else: # Choose random action. This is the epsilon case.
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        if (self.statesAdded == False): self.addStates(str(s))
        self.check_state_exist(s_)
        while (self.stable == False):
            self.policyEvaluation()
            self.policyImprovement()

        if s_ != 'terminal':
            a_ = self.pi[str(s)]
            self.V[s] = r + self.gamma * self.V[s_]
        else:
            self.V[s] = r  # next state is terminal
            # self.train = True
        return s_, a_

    def policyEvaluation(self):
        while True:
            delta = 0
            for s in list(self.pi.keys()):
                if (s in [str(self.env.canvas.coords(w)) for w in self.env.pitblocks]):
                    self.V[s] = 0
                    continue
                v, action = self.V[s], self.pi[s]
                s_, r, _ = step(self.env, s, action)
                self.V[s] = r + self.gamma * self.V[s_]
                delta = max(delta, np.abs(v-self.V[s]))
            if (delta < self.theta):
                return

    def policyImprovement(self):
        policy_stable = True
        for s in list(self.pi.keys()):
            old_action = self.pi[s]
            A = self.lookAhead(s)
            self.pi[s] = np.argmax(A)
            if self.pi[s] != old_action:
                policy_stable = False
        self.stable = policy_stable

    '''States are dynamically added to the tables as they are encountered'''
    def check_state_exist(self, s):
        if s not in self.pi:
            self.pi[str(s)] = np.random.choice(self.actions)
        if s not in self.V:
            self.V[str(s)] = 0

    def lookAhead(self, s):
        A = np.zeros(len(self.actions))
        for a in range(len(self.actions)):
            s_, reward, _ = step(env=self.env, state=str(s), action=a)
            # self.check_state_exist(state=s_)
            A[a] = reward + self.gamma * self.V[s_]
        return A

    def addStates(self, s):
        state = s
        down, right = 1, 2
        for _ in range(self.env.MAZE_H):
            temp_state = state
            for _ in range(self.env.MAZE_W):
                self.check_state_exist(temp_state)
                s_, _, _ = step(self.env, temp_state, right)
                temp_state = s_
            state_, _, _ = step(self.env, state, down)
            state = state_
        self.statesAdded = True