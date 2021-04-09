"""
Project for Udacity Danaodgree in Deep Reinforcement Learning

This script train an agent to navigate (and collect bananas!) in a large, square world.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.
The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
0 - move forward.
1 - move backward.
2 - turn left.
3 - turn right.
The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes. 
"""
from unityagents import UnityEnvironment
import numpy as np
from collections import deque
from dqn_agent import Agent
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
Unity environment configuration

Mac: "path/to/Banana.app"
Windows (x86): "path/to/Banana_Windows_x86/Banana.exe"
Windows (x86_64): "path/to/Banana_Windows_x86_64/Banana.exe"
Linux (x86): "path/to/Banana_Linux/Banana.x86"
Linux (x86_64): "path/to/Banana_Linux/Banana.x86_64"
Linux (x86, headless): "path/to/Banana_Linux_NoVis/Banana.x86"
Linux (x86_64, headless): "path/to/Banana_Linux_NoVis/Banana.x86_64"
"""
# start Unity environment
env = UnityEnvironment(file_name="Banana.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=False)[brain_name]

action_size = brain.vector_action_space_size
state_size = len(env_info.vector_observations[0])

# initialize agent
agent = Agent(state_size=state_size, action_size=action_size, seed=0, device=device)

def train(n_episodes=2000, eps_start=1.0, eps_end=0.05, eps_decay=0.99):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        # reset environment
        env_info = env.reset(train_mode=True)[brain_name]
        # get initial state
        state = env_info.vector_observations[0]
        # set initial score
        score = 0
        
        while True:
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            
            next_state, reward, done = env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=14:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

train()