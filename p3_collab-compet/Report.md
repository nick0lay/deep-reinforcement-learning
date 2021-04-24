[img_maddpg_performance]: img/img_maddpg_performance.png
[img_ddpg_performance]: img/img_ddpg_performance.png

# Report

In this project DDPG (Deep Deterministic Policy Gradient) and MADDPG (Multi-Agent Deep Deterministic Policy Gradient) architecures was used.

1. DDPG architecture impmlemented in `Tennis.ipynb`
2. MADDPG architecture implemented in `Tennis-MADDPG.ipynb`

## Environment description
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,
 - After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
 - This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Networks architecture
Critic is a fully connected network with 2 hidden layers with 256 and 128 neurons. It maps (state, action) pairs into Q-values. Action is feeded after first hidden layer.

Actor is a fully connected neural network with 2 hidden layers with 256 and 128 neurons. Input to the netowrk is the state vector and output is the action values.

Networks architectures implemented in `model.py`.

Critic network parameters:
 - state_size (int): Dimension of each state
 - action_size (int): Dimension of each action
 - seed (int): Random seed
 - fcs1_units (int): Number of nodes in the first hidden layer
 - fc2_units (int): Number of nodes in the second hidden layer
 - use_batch_norm: Flag to enable or disable batch normalisation
 - use_dropout: Flag to enable or disable dropout
 
Critic selected parameters:
 - state_size = 8 (continuous)
 - action_size = 2 (continuous)
 - fcs1_units = 256
 - fc2_units = 128
 - seed = 0
 - use_batch_norm = True
 - use_dropout = False
 
Actor network parameters:
 - state_size (int): Dimension of each state
 - action_size (int): Dimension of each action
 - seed (int): Random seed
 - fc1_units (int): Number of nodes in first hidden layer
 - fc2_units (int): Number of nodes in second hidden layer
 - use_batch_norm: Flag to enable or disable batch normalisation
 - use_dropout: Flag to enable or disable dropout
 
Actor selected parameters:
 - state_size = 8 (continuous)
 - action_size = 2 (continuous)
 - fc1_units = 256
 - fc2_units = 128
 - seed = 0
 - use_batch_norm = True
 - use_dropout = False

Enabling batch normalisation reduce number of episodes required to solve environment up to 2 times.

Dropout can be enabled to avoid network overfitting if required. For current implementation with selected parameters above it's not required.

## Learning Algorithm

### DDPG agent
Agent implemented in `ddpg_agent.py`. This solution use single DDPG agent to train both environment agents. Environment agents use the same critic and actor networks.

Agent parameters:
 - BUFFER_SIZE - replay buffer size
 - BATCH_SIZE - minibatch size
 - GAMMA - discount factor
 - TAU - for soft update of target parameters
 - LR_ACTOR - learning rate of the actor 
 - LR_CRITIC - learning rate of the critic
 - WEIGHT_DECAY - L2 weight decay
 - UPDATE_EVERY - update network frequency
 - LEARN_COUNT - learn network count

Selected parameters:
 - BUFFER_SIZE = int(1e5)
 - BATCH_SIZE = 512
 - GAMMA = 0.99
 - TAU = 1e-1
 - LR_ACTOR = 1e-4 
 - LR_CRITIC = 3e-4
 - WEIGHT_DECAY = 0.0
 - UPDATE_EVERY = 5
 - LEARN_COUNT = 10

#### MADDPG agent
Agent implemented in `maddpg_agent.py` which use two DDPG agents under the hood. DDPG agnet implemented in `agent.py`. This solution use two DDPG agents coordinated by MADDPG agent. DDPG agents use ther own actor network and shared critic network. Actor network trained only by own actor obserbvations but critic network use observations from both agents.

Agent parameters:
 - BUFFER_SIZE - replay buffer size
 - BATCH_SIZE - minibatch size
 - GAMMA - discount factor
 - TAU - for soft update of target parameters
 - LR_ACTOR - learning rate of the actor 
 - LR_CRITIC - learning rate of the critic
 - WEIGHT_DECAY - L2 weight decay
 - UPDATE_EVERY - update network frequency
 - LEARN_COUNT - learn network count
 
Selected parameters:
 - BUFFER_SIZE = int(1e5)
 - BATCH_SIZE = 512
 - GAMMA = 0.99
 - TAU = 1e-1
 - LR_ACTOR = 1e-4 
 - LR_CRITIC = 3e-4
 - WEIGHT_DECAY = 0.0
 - UPDATE_EVERY = 5
 - LEARN_COUNT = 10

### Algorithm Performance
#### DDPG agent
![Algorithm Performance][img_ddpg_performance]

Environment was solved in 404 episodes with average score 0.5.
```
Episode 17	Average Score: 0.00
Episode 100	Average Score: 0.03
Episode 200	Average Score: 0.08
Episode 300	Average Score: 0.13
Episode 400	Average Score: 0.48
Episode 404	Average Score: 0.50
Environment solved in 304 episodes!	Average Score: 0.50
```

#### MADDPG agent
![Algorithm Performance][img_maddpg_performance]

Environment was solved in 392 episodes with average score 0.5.
```
Episode 31	Average Score: 0.01
Episode 100	Average Score: 0.04
Episode 200	Average Score: 0.08
Episode 300	Average Score: 0.12
Episode 392	Average Score: 0.50
Environment solved in 292 episodes!	Average Score: 0.50
```

### Noise algorithm improvements
Noise algorithm implemented in `OUNoise` class and implement Ornstein-Uhlenbeck process. Since original implementation generate only positive noise in most cases it moves an agent towards the net. This movement was quite inefficient for environment discovery especially at the training start. As an improvement this class was modified to generate positive and negative noise.

### Future Improvements
Try to use Trust Region Optimization (TRPO) and Proximal Policy Optimization (PPO) algorithms to solving this environment.

Play with hyper parameters to improve algorithm performance and stability.