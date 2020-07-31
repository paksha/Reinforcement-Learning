import argparse
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from maze_env import Maze

wall_shape=np.array([[6,3],[6,3],[6,2],[5,2],[4,2],[3,2],[3,3],
        [3,4],[3,5],[3,6],[4,6],[5,6],[5,7],[7,3]])
pits=np.array([[1,3],[0,5], [7,7], [8,5]])
agentXY=[0,0]
goalXY=[4,4]

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = Maze(agentXY,goalXY,wall_shape, pits)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(100, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.affine2 = nn.Linear(128, 4)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-6)
eps = np.finfo(np.float32).eps.item()

def decode(state):
    new_state = state[:2]
    new_state = [int(new_state[0]-5)//40, int(new_state[1]-5)//40]
    cell = 10*new_state[1]+new_state[0]
    e = np.zeros(100)
    e[cell] = 1
    return e

def select_action(state):
    state = torch.from_numpy(np.array(decode(state))).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    eps = []
    points = []
    for i_episode in range(20000):
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done = env.step(action)
            # if args.render:
            #     env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                eps.append(i_episode)
                points.append(ep_reward)
                if i_episode % 500 == 0:
                    plt.plot(eps, points)
                    plt.show()
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break


if __name__ == '__main__':
    main()