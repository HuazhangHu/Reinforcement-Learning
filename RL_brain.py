'''
author ambition
date 19.7.22
function 
version
'''

import numpy as np
import pandas as pd


class RL:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy#epsilon用于判断是选择收益最大的还是选择冒险，防止陷入局部最优
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    # def learn(self, s, a, r, s_):
    #     self.check_state_exist(s_)
    #     q_predict = self.q_table.loc[s, a]
    #     if s_ != 'terminal':
    #         q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
    #         #Q-learning选择最有可能的动作
    #     else:
    #         q_target = r  # next state is terminal
    #     self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  #更新Q表


    def check_state_exist(self, state):
        #if exist a new state,append it
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

#off-policy
class QLearningTable(RL):
    #继承父类并重载
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
            #选择可能性最大的state及其action(选择最大的)
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update


# on-policy
class SarsaTable(RL):
    #继承父类RL并重载
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):#重载
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
            #sarsa的区别之处在于他直接选择进入下一个state和下一个action(基于epsilon选择)
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # 更新Q表