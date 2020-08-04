from maze_env import Maze
from RL_brainsample_PI import rlalgorithm as rlalg1
from RL_PolicyIteration import AsyncPolicyIteration as asyncPI
from RL_ValueIteration import AsyncValueIteration as asyncVI
from RL_Sarsa import Sarsa
from RL_QLearning import QLearning
from RL_ExpectedSarsa import ExpectedSarsa as expSarsa
from RL_Double_QLearning import DoubleQLearning as dql
from RL_Sarsa_ET import SarsaET
from pytorchReinforce import Policy as pg
import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

DEBUG=1
MAX_STEPS = 10000
def debug(debuglevel, msg, **kwargs):
    if debuglevel <= DEBUG:
        if 'printNow' in kwargs:
            if kwargs['printNow']:
                print(msg)
        else:
            print(msg)


def plot_rewards(experiments):
    color_list=['blue','green','red','black','magenta']
    label_list=[]
    for i, (env, RL, data) in enumerate(experiments):
        x_values=range(len(data['global_reward']))
        label_list.append(RL.display_name)
        y_values=data['global_reward']
        plt.plot(x_values, y_values, c=color_list[i],label=label_list[-1])
        plt.legend(label_list)
    plt.title("Reward Progress", fontsize=24)
    plt.xlabel("Episode", fontsize=18)
    plt.ylabel("Return", fontsize=18)
    plt.tick_params(axis='both', which='major',
                    labelsize=14)
    plt.show()

def update(env, RL, data, episodes=50):
    global_reward = np.zeros(episodes)
    data['global_reward']=global_reward
    start = datetime.now()
    for episode in range(episodes):
        t=0
        # initial state
        if episode == 0:
            state = env.reset(value = 0)
        else:
            state = env.reset()

        debug(2,'state(ep:{},t:{})={}'.format(episode, t, state))

        # RL choose action based on state
        action = RL.choose_action(str(state))
        while True:
            # fresh env
            if(showRender or (episode % renderEveryNth)==0):
                env.render(sim_speed)

            # RL take action and get next state and reward
            state_, reward, done = env.step(action)
            global_reward[episode] += reward
            if RL.display_name == "Policy Gradient":
                RL.rewards.append(reward)
            debug(2,'state(ep:{},t:{})={}'.format(episode, t, state))
            debug(2,'reward_{}=  total return_t ={} Mean50={}'.format(reward, global_reward[episode],np.mean(global_reward[-50:])))

            # RL learn from this transition
            # and determine next state and action
            if RL.display_name != "Policy Gradient": 
                state, action =  RL.learn(str(state), action, reward, str(state_))
            else:
                state, action = state_, RL.choose_action(str(state_))
                

            # break while loop when end of this episode
            if done or t > MAX_STEPS: 
                # RL.eps_decay() # THIS WILL BREAK IF THE RL ALGORITHM DOESN'T HAVE EPSILON DECAY
                if RL.display_name == "Policy Gradient":
                    RL.eps.append(episode)
                    RL.learn()
                break
            
            t += 1

        debug(1,"({}) Episode {}: Length={}  Total return = {} ".format(RL.display_name,episode, t,  global_reward[episode],global_reward[episode]),printNow=(episode%printEveryNth==0))
        if(episode>=100):
            debug(1,"    Median100={} Variance100={}".format(np.median(global_reward[episode-100:episode]),np.var(global_reward[episode-100:episode])),printNow=(episode%printEveryNth==0))
    # end of game
    end = datetime.now()
    time_taken = end - start
    print('Time: ',time_taken)  
    print('game over -- Algorithm {} completed'.format(RL.display_name))
    env.destroy()

def runExperiments(wall_shape, pits, episodes):
    
    env1 = Maze(agentXY,goalXY,wall_shape, pits)
    RL1 = pg()
    data1={}
    env1.after(10, update(env1, RL1, data1, episodes))
    env1.mainloop()
    experiments = [(env1,RL1, data1)]
    
    env2 = Maze(agentXY,goalXY,wall_shape,pits)
    RL2 = dql(actions=list(range(env1.n_actions)))
    data2={}
    env2.after(10, update(env2, RL2, data2, episodes))
    env2.mainloop()
    experiments.append((env2,RL2, data2))

    # Note that the Model-free methods have a different constructor
    # Only pass in the environment actions, not the environment itself unlike above

    env3 = Maze(agentXY,goalXY,wall_shape,pits)
    RL3 = expSarsa(actions=list(range(env3.n_actions)))
    data3={}
    env3.after(10, update(env3, RL3, data3, episodes))
    env3.mainloop()
    experiments.append((env3,RL3, data3))

    env4 = Maze(agentXY,goalXY,wall_shape,pits)
    RL4 = SarsaET(actions=list(range(env4.n_actions)))
    data4={}
    env4.after(10, update(env4, RL4, data4, episodes))
    env4.mainloop()
    experiments.append((env4,RL4, data4))
    
    return experiments



if __name__ == "__main__":
    sim_speed = 0

    showRender=False
    episodes=100
    renderEveryNth=5000
    printEveryNth=200
    do_plot_rewards=True

    if(len(sys.argv)>1):
        episodes = int(sys.argv[1])
    if(len(sys.argv)>2):
        showRender = sys.argv[2] in ['true','True','T','t']
    if(len(sys.argv)>3):
        datafile = sys.argv[3]


    # Task Specifications

    agentXY=[0,0]
    goalXY=[4,4]

    # Task 1
    #wall_shape=np.array([[2,2],[3,6]])
    #pits=np.array([[6,3],[1,4]])

    # To run all algorithms on Task 1, uncomment the following line
    # experiments = runExperiments(wall_shape, pits, episodes)

    # Task 2
    #wall_shape=np.array([[6,2],[5,2],[4,2],[3,2],[2,2],[6,3],[6,4],[6,5],[2,3],[2,4],[2,5]])
    #pits=[]

    # To run all algorithms on Task 2, uncomment the following line
    # experiments = runExperiments(wall_shape, pits, episodes)

    # Task 3
    wall_shape=np.array([[6,3],[6,3],[6,2],[5,2],[4,2],[3,2],[3,3],[3,4],[3,5],[3,6],[4,6],[5,6],[5,7],[7,3]])
    pits=np.array([[1,3],[0,5], [7,7], [8,5]])

    # To run all algorithms on Task 3, uncomment the following line
    experiments = runExperiments(wall_shape, pits, episodes)
    
    print("All experiments complete")

    for env, RL, data in experiments:
        print("{} : max reward = {} medLast100={} varLast100={}".format(RL.display_name, np.max(data['global_reward']),np.median(data['global_reward'][-100:]), np.var(data['global_reward'][-100:])))


    if(do_plot_rewards):
        #Simple plot of return for each episode and algorithm, you can make more informative plots
        plot_rewards(experiments)
    #Not implemented yet
    #if(do_save_data):
    #    for env, RL, data in experiments:
    #        saveData(env,RL,data)

