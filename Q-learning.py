'''
author ambition
date 19.7.22
function some demo of  Q-learning,reinforce learning
version
'''

import numpy as np
import pandas as pd
import time

np.random.seed(2)  # reproducible


N_STATES = 16   # 游戏总长
ACTIONS = ['left', 'right'] # available actions
EPSILON = 0.9   # greedy police 0.9选择最有可能的,0.1随机选择
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 100   # 最大回合数
FRESH_TIME = 0.3    # fresh time for one move


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    print(table)    # show table
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    # print(state_actions)
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):#非贪婪选择或者这个state还未探索过
        action_name = np.random.choice(ACTIONS)#随机去选择
    else:   # act greedy
        action_name = state_actions.idxmax()#选择最大可能的
    return action_name


def get_env_feedback(S, A):
    #环境反馈,S代表当前位置.S_代表下一个状态
    if A == 'right':    # move right
        if S == N_STATES - 2:   # terminate
            S_ = 'terminal'#下一步就到了（从0开始,所以是减2)
            R = 1#奖励为1
        else:
            S_ = S + 1
            R = 0
    else:   # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):
    #跟新环境
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 6#初始位置
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)#环境反馈
            q_predict = q_table.loc[S, A]#Q预期
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()#奖励
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # 游戏继续

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # 跟新Q表
            S = S_  # move to next state

            update_env(S, episode, step_counter+1)
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)


