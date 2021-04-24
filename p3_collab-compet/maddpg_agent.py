from agent import Agent, ReplayBuffer
import torch
import numpy as np

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
UPDATE_EVERY = 5
LEARN_EVERY = 10
GAMMA = 0.99

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG:
    def __init__(self, num_agents, state_size, action_size, random_seed = 0, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()

#         self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.memories = [ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed) for _ in range(num_agents)]
        self.agents = [Agent(state_size, action_size, random_seed) for _ in range(num_agents)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
    
    def reset(self):
        for agent in self.agents:
            agent.reset()
            
    def act(self, states, add_noise=True):
        actions = [agent.act(state, add_noise) for agent, state in zip(self.agents, states)]
        return actions
    
    def step(self, time_step, states, actions, rewards, next_states, dones):
        for state, action, reward, next_state, done, memory in zip(states, actions, rewards, next_states, dones, self.memories):
            memory.add(state, action, reward, next_state, done)
        
        if time_step % UPDATE_EVERY != 0:
            return
        
        # Learn, if enough samples are available in memory
#         if len(self.memory) > BATCH_SIZE:
        memory_redines = [len(memory) > BATCH_SIZE for memory in self.memories]
        if np.all(memory_redines):
            for i in range(LEARN_EVERY):
                for agent, memory in zip(self.agents, self.memories):
                    experiences = memory.sample()
                    agent.learn(experiences, GAMMA)
#                 for agent in self.agents:
#                     experiences = self.memory.sample()
#                     agent.learn(experiences, GAMMA)

#     def step(self, time_step, states, actions, rewards, next_states, dones):
#         for agent, state, action, reward, next_state, done in zip(self.agents, states, actions, rewards, next_states, dones):
#             agent.step(time_step, states, action, reward, next_state, done)