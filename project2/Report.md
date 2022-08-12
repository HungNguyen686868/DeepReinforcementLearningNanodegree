# Banana Project Report

## Introduction

In this project, I will train an agent to navigate (and collect bananas!) in a large, square world.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

0 - move forward.
1 - move backward.
2 - turn left.
3 - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

## Solutions
In this project, I applied the Double DQN incoporating with Experience Repalay to train an agent. The details are given as follows:

* The standard DQN method has been shown to overestimate the true Q-value, because for the target an argmax over estimated Q-values is used. Double DQN utilises Double Q-learning to reduce overestimation by decomposing the max operation in the target into action selection and action evaluation. By using two uncorralated Q-Networks we can prevent this overestimation. We evaluate the greedy policy according to the local network, but we use the target network to estimate its value. The update is the same as for DQN, but replacing the target $Y^{DQN}$ with:    $Y^{DDQN} = R_{t+1} + \lambda Q (S_{t+1} + argmax_{a} Q (S_{t+1}, a, \theta_t); \theta^{-}_t)$ . 
* The loss function is given by:  $(Q(S_t, A_t) - Q_{\theta}(S_t, A_t))^2$
### 1. DQNetwork

The DQNetwork is implemented in Pytorch. This network containts 04 fully connected layers (each layer have 64 units) and Relu activation function.
The hyperparameters are choosen from many attempts "trail and error" based on training sessions on the CPU. These hyperparameters are given as follows:

* lr = 0.0001           ( learning rate) 
* epsilon = 1.0, 
* eps_dec = 1e-5, 
* eps_min = 0.01, 
* buffer_size = 10000,  (replay buffer)
* batch_size = 32,      (batch size)
* update = 4,           (frequence to update the network)
* tau = 0.01            (for soft update the target network's paramaters)
### 2. Replay Buffer 
I create a ReplayBuffer that stores the last buffer_size S.A.R.S. (State, Action, Reward, New State) experiences.  During training, the agent will be trained on  #batch_size experiences (randomly sampling from the replay buffer). The agen will accumulate experiences through the replay-buffer until it (the buffer) has at least #batch_size experiences. 

Using replay memory allows to break the correlation between consecutive samples and help to improve the performance of the model. If the network learned only from consecutive samples of experience as they occurred sequentially in the environment, the samples would be highly correlated and would therefore lead to inefficient learning. Taking random samples from replay memory breaks this correlation.
## Results
The agent is trained over 1000 episodes and acheived a high score. It is able to get an average score of +13 over 100 consecutive episodes.
The training log can be found as beblow:

Episode 0	Average Score: 1.00

Episode 100	Average Score: 5.66

Episode 200	Average Score: 11.06

Episode 300	Average Score: 10.93

Episode 400	Average Score: 11.77

Episode 500	Average Score: 13.45

Episode 600	Average Score: 13.08

Episode 700	Average Score: 13.24

Episode 800	Average Score: 13.64

Episode 900	Average Score: 13.77

Episode 999	Average Score: 13.85
