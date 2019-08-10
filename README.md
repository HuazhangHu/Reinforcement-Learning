# Reinforcement-Learning
Single-agent Reinforcement Learning and Multi-agent Reinforcement Learning
sourse movan python(add some notes)
### Q-learning
value-based  
off-policy  
### Sarsa
value-based  
on-policy
### DQN
It is the conbine of Q-learning and Deep learning.  
value-based,of-policy.  
the critical step:memory replay and fixed target-network.  
### Policy Gradient 
on-policy,not value-based      
input: observations  
labels:actions  
output:probability    
Loss=cross entropy(layer out,labels)  
reward decay:the oldest data,the decay less
