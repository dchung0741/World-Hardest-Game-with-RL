from numpy import array, mean
import torch 
from torch import FloatTensor, Tensor
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

import gym

import matplotlib.pyplot as plt

# REF: Explicit algorithm is found in https://www.informit.com/articles/article.aspx?p=2995356&seqNum=3
PATH = 'WorldHardestGame_A2C_Model/'



class Policy_Net(nn.Module):

    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    
    def forward(self, state):

        return self.net(state)


class Actor:

    def __init__(self, env) -> None:
        
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n

        self.net = Policy_Net(self.obs_dim, self.act_dim)
        self.optim = optim.Adam(self.net.parameters(), lr = 1e-3)
        
        self.log_prob = []


    def choose_action(self, state):

        state = FloatTensor(state)
        policy_net_out = self.net(state).softmax(-1)
        
        dist = Categorical(policy_net_out)
        
        action = dist.sample(torch.Size([1]))
        log_prob = dist.log_prob(action)
        
        self.log_prob.append(log_prob)

        return action.item()

    
    def learn(self, td_error):
        
        self.log_prob = torch.cat(self.log_prob)
        td_error = td_error.detach()

        # Compute loss
        actor_loss = - td_error * self.log_prob
        actor_loss = actor_loss.mean()

        # Update policy net
        self.optim.zero_grad()
        actor_loss.backward()
        self.optim.step()
        
        # Reset actor's memory 
        self.log_prob = [] 
    



class Value_Net(nn.Module):

    def __init__(self, in_dim) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    
    def forward(self, state):
        state = FloatTensor(state)
        return self.net(state)


class Critic:

    def __init__(self, env) -> None:
        
        self.obs_dim = env.observation_space.shape[0]

        self.net = Value_Net(self.obs_dim)
        self.optim = optim.Adam(self.net.parameters(), lr = 1e-3)


class A2C:

    def __init__(self, env, n_accumulate) -> None:

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n

        self.actor = Actor(self.env)        
        self.critic = Critic(self.env)


        self.n = n_accumulate
        self.gamma = 0.99


    def train(self):
        
        state = self.env.reset()
        t = 0

        reward_list = []
        disc_reward_list = []
        state_value_list = []

        # Generate an episode
        for _ in range(self.env.spec.max_episode_steps):
            
            action = self.actor.choose_action(state)
            new_state, reward, done, _ = self.env.step(action)

            if len(state_value_list) == 0:
                state_value = self.critic.net(state)
                state_value_list.append(state_value)

            next_state_value = self.critic.net(new_state)            
            state_value_list.append(next_state_value)

            reward_list.append(reward)

            # Compute n-step discounted reward
            if len(disc_reward_list) != 0:
                
                first_idx = max(0, t - self.n)
                last_idx = len(disc_reward_list) - 1
                max_pow = min(self.n - 1, last_idx - first_idx + 1)
                
                #print('------------------------------------------------------------------------------------------------')
                for pow, i in zip(reversed(range(1, max_pow + 1)), range(first_idx, last_idx + 1)):
                    
                    #print(f'{disc_reward_list[i]} + {reward} * {self.gamma}^{pow}')
                    disc_reward_list[i] += reward * self.gamma**pow

            disc_reward_list.append(reward)

            state = new_state

            self.env.render()
            
            t += 1

            if done:
                break
        

        reward_list = FloatTensor(reward_list)
        disc_reward_list = FloatTensor(disc_reward_list)        
        state_value_list = torch.cat(state_value_list)

        val_tar_list = disc_reward_list + self.gamma**self.n * torch.cat((state_value_list[self.n:], FloatTensor([0. for _ in range(self.n - 1)])))

        value_loss = (val_tar_list - state_value_list[:-1])**2
        value_loss = value_loss.mean()
        advantage = (val_tar_list - state_value_list[:-1]).detach()

        self.critic.optim.zero_grad()
        value_loss.backward()
        self.critic.optim.step()

        self.actor.learn(advantage)


        return t, sum(reward_list)



def training_loop(env):

    a2c = A2C(env, n_accumulate = 5)

    total_episode = 1000
    episode_steps = []
    episode_reward = []
    
    for eps in range(total_episode):

        eps_step, eps_reward = a2c.train()
        episode_steps.append(eps_step)
        episode_reward.append(eps_reward)
        
        if len(episode_reward) >= 2:
            if episode_reward[-1] > episode_reward[-2]:
                torch.save(a2c.actor.net.state_dict(), PATH + 'Actor_Net_Model.pt')
                torch.save(a2c.critic.net.state_dict(), PATH + 'Critic_Net_Model.pt')

        print(f'\rEpisode {eps}, Step: {episode_steps[-1]:.2f}, Average Steps: {mean(episode_steps[-10::]):.2f}, Score: {episode_reward[-1]:.2f}, Average Score: {mean(episode_reward[-10::]):.2f},  Best Score: {max(episode_reward)}', end="")

        if eps % 100 == 0:
             print(f'\rEpisode {eps},  Step: {episode_steps[-1]:.2f}, Average Steps: {mean(episode_steps[-10::]):.2f}, Score: {episode_reward[-1]:.2f}, Average Score: {mean(episode_reward[-10::]):.2f}, Best Score: {max(episode_reward)}')

    
    episode_reward_moving_avg = [episode_reward[:i][-10:] for i in range(len(episode_reward))]
    episode_reward_moving_avg = list(map(mean, episode_reward_moving_avg))
    #plt.plot(episode_reward)
    plt.plot(episode_reward_moving_avg)
    plt.show()


if __name__ == '__main__':

    import World_Hardest_Game

    env = gym.make('CartPole-v1')
    env = gym.make('LunarLander-v2')
    env = gym.make('WorldHardestGame-v1')

    training_loop(env)
    
    
    



