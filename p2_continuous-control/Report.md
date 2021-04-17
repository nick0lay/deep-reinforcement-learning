[img_algorithm_performance]: images/algorithm_performance.png

# Report

In this prlject DDPG (Deep Deterministic Policy Gradient) architecure was used.

## Environment description
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## Learning Algorithm

Implementation consist of:
 - dqn_agent.py - agent implementation
 - model.py - neural network implementation

Agent parameters:
 - state_size - agent state size
 - action_size - agent action size
 
Algorithm parameters:
 - n_episodes - number of episodes to train model
 - BUFFER_SIZE - replay buffer size
 - BATCH_SIZE - minibatch size
 - GAMMA - discount factor
 - TAU - for soft update of target parameters
 - LR_ACTOR - learning rate of the actor 
 - LR_CRITIC - learning rate of the critic
 - WEIGHT_DECAY - L2 weight decay

Network parameters:
 - state_size - model input size
 - action_size - model output size
 - fc1_units - first fully connected layer size
 - fc2_units - second fully connected layer size
 
Selected parameters:
 - state_size = 33
 - action_size = 4
 - n_episodes = 300
 - BUFFER_SIZE = 1e6
 - BATCH_SIZE = 128
 - GAMMA = 0.99
 - TAU = 1e-3
 - LR_ACTOR = 1e-4
 - LR_CRITIC = 1e-4
 - WEIGHT_DECAY = 0
 - fc1_units = 256
 - fc2_units = 128

### Algorithm Performance
![Algorithm Performance][img_algorithm_performance]

Environment was solved in 135 episodes with average score 30.06.
```
Episode 100	Average Score: 9.23
Episode 200	Average Score: 24.88
Episode 235	Average Score: 30.06
Environment solved in 135 episodes!	Average Score: 30.06
```

### Future Improvements
Explor Proximal Policy Optimization (PPO) and Distributed Distributional Deterministic Policy Gradients (D4PG) algorithms to solving this environment.

Play with hyper parameters to improve algorithm performance and stability.