# Continous Control Project Report

## Introduction

In this project, I trained an agent to maintain its position at the target location for as many time steps as possible. In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

I choose to solve the First Version that contains a single agent. The task is episodic, and in order to solve the environment, an agent must get an average score of +30 over 100 consecutive episodes.

## Solutions
In this project, I applied the DDPG algorithm to train an agent. The details about DDPG are given as follows:
* DDPG, or Deep Deterministic Policy Gradient, is an actor-critic, model-free algorithm based on the deterministic policy gradient that can operate over continuous action spaces. It combines the actor-critic approach with insights from DQNs: in particular, the insights that 1) the network is trained off-policy with samples from a replay buffer to minimize correlations between samples, and 2) the network is trained with a target Q network to give consistent targets during temporal difference backups

I adapt the code from the Udacity's example [ddpg-pendulum](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) and [ShangtongZhang's github repo](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction) to this project by making as few modifications as possible.

- Using Elu activation instead of Relu activation
- Using bigger Actor-Critic networks
- Using Gradient Clipping to prevent gradient vanishing/exploring

### 1. Actor-Critic networks

The Actor and CriticsNetworks are implemented in Pytorch. The Actor network containts 03 fully connected layers (512- 512 -4 units), 02 BatchNorm layers and Elu activation function. 

```python
class ActorNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_dims = 512, fc2_dims = 512):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_dims (int): Number of nodes in first hidden layer
            fc2_dims (int): Number of nodes in second hidden layer            
            lr (float): learning rate
        """
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.bn1 = nn.BatchNorm1d(fc1_dims)
        self.bn2 = nn.BatchNorm1d(fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, action_size) 
```

The Critic network containts 04 fully connected layers (with 512-1024-512-1 units) and Elu activation function.

```python
class CriticNetwork(nn.Module):
    """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

    def __init__(self, state_size, action_size, seed, fc1_dims = 512, fc2_dims = 512):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_dims (int): Number of nodes in first hidden layer
            fc2_dims (int): Number of nodes in second hidden layer
            fc3_dims (int): Number of nodes in 3rd hidden layer
            fc4_dims (int): Number of nodes in 4th hidden layer
            lr (float): learning rate
        """
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_dims)
        self.fc2 = nn.Linear(action_size+ fc1_dims, 2*fc2_dims)
        self.fc3  = nn.Linear(2*fc2_dims , fc2_dims)  
        self.fc4 =  nn.Linear(fc2_dims , 1)
```

The hyperparameters are choosen from many attempts "trail and error" based on training sessions on the workplace's GPU provided by Udacity. These hyperparameters are given as follows:

```python
config = {"buffer_size": 1000000,  # replay buffer size
          "batch_size" : 128,        # minibatch size
          "gamma" : 0.99,            # discount factor
          "tau" : 1e-3,              # for soft update of target parameters
          "lr_actor" : 3e-4,         # learning rate of the actor 
          "lr_critic": 4e-4,        # learning rate of the critic
          "update": 4               # Update times of critic/actor in each trajectory
             }
n_games = 1000  # number of trajectories
max_t = 1000000 # Max timestep in a trajectory
```
### 2. Replay Buffer 
I create a ReplayBuffer that stores the last buffer_size S.A.R.S. (State, Action, Reward, New State) experiences.  During training, the agent will be trained on  #batch_size experiences (randomly sampling from the replay buffer). The agen will accumulate experiences through the replay-buffer until it (the buffer) has at least #batch_size experiences. 

Using replay memory allows to break the correlation between consecutive samples and help to improve the performance of the model. If the network learned only from consecutive samples of experience as they occurred sequentially in the environment, the samples would be highly correlated and would therefore lead to inefficient learning. Taking random samples from replay memory breaks this correlation.

### 3. Gradient clipping
In this project, I apply [Gradient Clipping](https://paperswithcode.com/method/gradient-clipping) to train Actor-Critic Network. Actually, Gradient Clipping clips the size of the gradients to ensure optimization performs more reasonably near sharp areas of the loss surface. It can be performed in a number of ways. One option is to simply clip the parameter gradient element-wise before a parameter update. Another option is to clip the norm ||g || of the gradient  before a parameter update:
if $\lVert g\lVert > v$ then  $g \leftarrow \frac{gv}{\lVert g\lVert}$    where  $v$ is a norm threshold.

In pytorch, I use ```python torch.nn.utils.clip_grad_norm_()```, Clips gradient norm of an iterable of parameters. The norm is computed over all gradients together, as if they were concatenated into a single vector. Gradients are modified in-place.

## Results
The agent is trained over 1000 episodes and acheived a high score. After 200 epiodes of training, the agent is able to get an average score of +35 rewards over 100 consecutive episodes. The training log can be found as beblow:


