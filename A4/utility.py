import numpy as np

'''step - definition of one-step dynamics function'''
def step(env, state, action):
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

    s_ =  move(s, base_action)

    # call the reward function
    reward, done, reverse = env.computeReward(s, action, s_)
    return str(s_), reward, done

'''Take the action from s and change the coordinates'''
def move(s, action):
    s[0] += action[0] # update x0
    s[1] += action[1] # update y0
    s[2] += action[0] # update x1
    s[3] += action[1] # update y1
    return s

def decode(state):
    new_state = state[:2]
    new_state = [int(new_state[0]-5)//40, int(new_state[1]-5)//40]
    cell = 10*new_state[1]+new_state[0]
    e = np.zeros(100)
    e[cell] = 1
    return e