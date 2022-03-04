import torch
import torch.nn as nn
import torch.utils as utils
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

import gym

import numpy as np
from numpy import array, log10
from numpy.random import choice
from numpy import mean

import matplotlib.pyplot as plt





class Policy_Network(nn.Module):

    def __init__(self, in_dim, out_dim) -> None:

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
        self.train()
    
    def forward(self, x):

        return self.network(x)#.squeeze(0)


class Actor:

    def __init__(self, env) -> None:
        
        self.env = env
        self.action_dim = self.env.action_space.n
        self.observation_space_dim = self.env.observation_space.shape[0]

        self.network = Policy_Network(self.observation_space_dim, self.action_dim)

        self.reward_memory = []
        self.log_prob_memory = []

        
        self.optim = optim.Adam(self.network.parameters(), lr = 1e-3)


    def choose_action(self, observation):

        observation = torch.FloatTensor(observation)

        
        #prob_dist = F.softmax(self.network(observation), dim = 0).detach()
        #prob_dist = Categorical(prob_dist)
        out = self.network.forward(observation)
        prob_dist = Categorical(logits = out)
        
        chosen_action = prob_dist.sample()
        self.log_prob_memory.append(prob_dist.log_prob(chosen_action))

        return chosen_action.item()

        
    def reset_memory(self):

        self.reward_memory = []
        self.log_prob_memory = []

    
    def learn(self, gamma = 0.99):

        state = env.reset()
        done = False
        
        # generate an episode
        for _ in range(self.env.spec.max_episode_steps):
            
            action = self.choose_action(state)
            new_state, step_reward, done, _ = env.step(action)
            self.reward_memory.append(step_reward)
            state = new_state
            env.render()

            if done:
                break
            


        # Compute R for each time step
        G_list = []

        for r in reversed(self.reward_memory):
            
            try:    
                G_list.append(r + G_list[-1] * gamma)
            
            except IndexError:
                G_list.append(r)

        
        G_list = G_list[::-1]
        G_list = torch.Tensor(G_list)
        log_prob_tensor = torch.stack(self.log_prob_memory)
        

        loss = log_prob_tensor * G_list
        loss = - torch.sum(loss)
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        return loss


def training_loop(env):

    actor = Actor(env = env)
    total_episode = 300

    loss_list = []
    eps_step_list = []
    eps_reward_list = []
    
    for eps in range(total_episode):
            
        loss = actor.learn()
        loss_list.append(loss)
        eps_step_list.append(len(actor.reward_memory))
        eps_reward_list.append(sum(actor.reward_memory))

        print(f'\rEpisode {eps}, Step: {eps_step_list[-1]}, Average Steps: {mean(eps_step_list[-10::]):.2f}, Score: {eps_reward_list[-1]:.2f}, Average Score: {mean(eps_reward_list[-10::]):.2f}', end="")

        if eps % 100 == 0:
             print(f'\rEpisode {eps}, Step: {eps_step_list[-1]}, Average Steps: {mean(eps_step_list[-10::]):.2f}, Score: {eps_reward_list[-1]:.2f}, Average Score: {mean(eps_reward_list[-10::]):.2f}')

        actor.reset_memory()
        actor.env.render()

    print('\n')

    return loss_list, eps_reward_list, eps_step_list


if __name__ == '__main__':

    import World_Hardest_Game

    env = gym.make('WorldHardestGame-v1')
    loss_list, eps_reward_list, eps_step_list = training_loop(env)
    env.close()
    #plt.plot(loss_list)
    plt.plot(eps_reward_list)
    plt.show()