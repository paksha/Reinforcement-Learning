import os
import sys
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from maze_env import Maze

class PolicyNetwork(nn.Module):

    def __init__(self, num_inputs=4, num_actions=4, hidden_size=16, learning_rate=1e-10, gamma=0.99):
        super(PolicyNetwork, self).__init__()

        self.gamma = gamma
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.display_name="REINFORCE"

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x

    def get_action(self, st):
        # state = hot_encode(str(st))
        state = np.array(st)
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob
    
def hot_encode(s):
    s = s.strip(",[]").split(", ")
    coords = [(int(i[:-2])-5)//40 for i in s]
    x, y = coords[0], coords[1]
    encoding = np.zeros(100)
    encoding[10*y+x] = 1
    return encoding

def update_policy(policy_network, rewards, log_probs):
    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0 
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + policy_network.gamma**pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)
        
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)
    
    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()

def main():
    wall_shape=np.array([[6,3],[6,3],[6,2],[5,2],[4,2],[3,2],[3,3],
        [3,4],[3,5],[3,6],[4,6],[5,6],[5,7],[7,3]])
    pits=np.array([[1,3],[0,5], [7,7], [8,5]])
    agentXY=[0,0]
    goalXY=[4,4]

    env = Maze(agentXY,goalXY,wall_shape, pits)
    policy_net = PolicyNetwork()
    
    max_episode_num = 5000
    max_steps = 1000
    numsteps = []
    avg_numsteps = []
    all_rewards = []

    for episode in range(max_episode_num):
        state = env.reset()
        log_probs = []
        rewards = []

        for steps in range(max_steps):
            # env.render()
            action, log_prob = policy_net.get_action(state)
            new_state, reward, done = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            
            if done or steps == max_steps-1:
                update_policy(policy_net, rewards, log_probs)
                numsteps.append(steps)
                avg_numsteps.append(np.mean(numsteps[-10:]))
                all_rewards.append(np.sum(rewards))
                if episode % 50 == 0:
                    sys.stdout.write("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3),  np.round(np.mean(all_rewards[-10:]), decimals = 3), steps))
                break
            
            state = new_state
        
    plt.plot(numsteps)
    plt.plot(avg_numsteps)
    plt.xlabel('Episode')
    plt.show()

main()