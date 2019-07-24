'''
author ambition
date 19.7.23
function  Q_learning and Sarsa Demo
version
'''

from maze_env import Maze
from RL_brain import QLearningTable
from RL_brain import SarsaTable
from RL_brain import RL
from time import time

def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()
        action= SA.choose_action(str(observation))
        #action= QL.choose_action(str(observation))
        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL choose action based on observation
            action_= SA.choose_action(str(observation_))

            #observation_代表下一步
            # RL learn from this transition
            #Q-learning
            #
            # QL.learn(str(observation),action,reward,str(observation_))
            # Sarsa
            SA.learn(str(observation),action,reward,str(observation_),action_)

            # swap observation

            observation = observation_
            #sarsa直接选择相应的动作
            action =action_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    t0=time()
    env = Maze()
    SA = SarsaTable(actions=list(range(env.n_actions)))
    # QL=QLearningTable(actions=list(range(env.n_actions))
    env.after(100, update)
    env.mainloop()
    print("spend %d",round(time()-t0),'s')