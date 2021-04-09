[img_algorithm_performance]: images/algorithm_performance.png
[img_deep_q_learning_algorithm]: images/deep_q_learning_algorithm.jpeg
[img_dqn_algorithm]: images/dqn_algorithm.png

### Learning Algorithm
Deep Q-Learning Algorithm (DQN) was used to solve navigation problem.
![Deep Q-Learning Algorithm (DQN) schema][img_dqn_algorithm]
![Deep Q-Learning Algorithm (DQN)][img_deep_q_learning_algorithm]

Implementation consist of:
 - dqn_agent.py - agent implementation
 - model.py - neural network implementation
 - replay_buffer.py - agent reply buffer
 - train.py - script to train model
 
Agent parameters:
 - state_size - agent state size
 - action_size - agent action size
 
Algorithm parameters:
 - n_episodes - number of episodes to train model
 - eps_start - starting epsilon for epsilon-greedy selection
 - eps_end - minimum value of epsilon
 - eps_decay - epsilon decay parameter
 - LR - Q-Network learning rate
 - GAMMA - reward discount parameters
 - UPDATE_EVERY - rate on which network need to be updated
 - TAU - interpolation parameter to copy weights from local network to target
 - BUFFER_SIZE - size of replay memory buffer
 - BATCH_SIZE - replay memory batch size, used to update model

Q-Network parameters:
 - state_size - model input size
 - action_size - model output size
 - fc1_units - first fully connected layer size
 - fc2_units - second fully connected layer size
 

Selected parameters:
 - state_size = 37 (defined by Banana environment)
 - action_size = 4 (defined by Banana environment)
 - n_episodes = 2000 (default value, usually environment solved in 600 - 700 episodes)
 - eps_start = 1.0
 - eps_end = 0.05
 - eps_decay=0.99
 - LR = 5e-4
 - GAMMA = 0.99
 - UPDATE_EVERY = 4
 - TAU = 1e-3
 - BUFFER_SIZE = 1e5
 - BATCH_SIZE = 64 (recommended 32 - 256)
 - state_size = 37
 - action_size = 4
 - fc1_units = 256
 - fc2_units = 256

### Q-Network architecture
Network architecture consists of 3 layers:
 - input layer FC1: 37 (state size) nodes in, 256 nodes out
 - hidden layer FC2: 256 nodes in, 256 nodes out
 - output layer FC3: 256 nodes in, 4 (action size) out

### Algorithm Performance
![Algorithm Performance][img_algorithm_performance]

Environment was solved in 643 episodes with average score 14.00.

### Future Improvements
Following extensions can be applied to improve performance:
 - Double DQN
 - Prioritized Experience Replay
 - Dueling DQN