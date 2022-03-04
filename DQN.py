import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
from numpy.random import randint
from numpy import mean
import random
import matplotlib.pyplot as plt

#############################################################################################################################################

class Replay_Memory:

    def __init__(self) -> None:

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.next_state_memory = []

        self.memory = [self.state_memory, self.action_memory, self.reward_memory, self.next_state_memory]
        
        self.capacity = 100

    def push(self, state, action, reward, next_state):

        self.state_memory.append(torch.FloatTensor(state))
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        self.next_state_memory.append(torch.FloatTensor(next_state))
    

    def __len__(self):

        return len(self.state_memory)
    

    def sample(self, n):
        
        sampled_idx = random.sample(range(len(self)), n)

        sampled_state = []
        sampled_action = []
        sampled_reward = []
        sampled_next_state = []

        for idx in sampled_idx:

            sampled_state.append(self.state_memory[idx])
            sampled_action.append(self.action_memory[idx])
            sampled_reward.append(self.reward_memory[idx])
            sampled_next_state.append(self.next_state_memory[idx])
        
        
        return torch.stack(sampled_state), torch.LongTensor([sampled_action]).view(n,1), torch.FloatTensor(sampled_reward), torch.stack(sampled_next_state)




class Network(nn.Module):

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
    
    def forward(self, x_in):

        return self.network(x_in)#.squeeze(0)



class DQN_Algorithm:

    def __init__(self, env) -> None:
        
        self.env = env
        self.action_dim = self.env.action_space.n
        self.observation_dim = self.env.observation_space.shape[0]

        self.eval_network = Network(self.observation_dim, self.action_dim)
        self.target_network = Network(self.observation_dim, self.action_dim)

        self.memory = Replay_Memory()

        self.loss_fnc = nn.MSELoss(reduction = 'mean')
        self.optim = optim.Adam(self.eval_network.parameters(), lr = 5e-3)


    def choose_action(self, state):

        if randint(10) < 1:

            return randint(self.action_dim)
        
        else:

            state_tensor = torch.FloatTensor(state)
            out = self.eval_network.forward(state_tensor)

            return torch.argmax(out).item()
    


    def update_eval_net(self, n_sample = 100, gamma = 0.99):

        # Sample from memory
        sampled_states, sampled_actions, sampled_rewards, sampled_next = self.memory.sample(n_sample)

        # Compute q(s) value from eval net
        q_eval = self.eval_network(sampled_states)
        q_eval = q_eval.gather(1, sampled_actions)
        
        # Compute q(s') value from target net
        q_next = self.target_network(sampled_next).detach()

        # Compute q(s) value from q(s')_{target}
        q_target = sampled_rewards + gamma * q_next.max(1)[0]
        q_target = q_target.unsqueeze(1)


        # Train network
        loss = self.loss_fnc(q_eval, q_target)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss


        
    
    def episode_generator(self, copy_to_target = False, train = False):

        state = self.env.reset()
        step = 0
        eps_reward = 0

        for _ in range(self.env.spec.max_episode_steps):
            
            step += 1
            
            action = self.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)
            
            # Modify reward for cartpole environment
            if str(self.env) == '<TimeLimit<CartPoleEnv<CartPole-v0>>>':
                x, _, theta, _ = next_state
                r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8 
                r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                reward = r1 + r2

            eps_reward += reward

            self.memory.push(state, action, reward, next_state)

            if train:
                loss = self.update_eval_net()
                

            if done:
                break
            else:
                state = next_state
                self.env.render()


        # copy weights in 
        if copy_to_target:
            self.target_network.load_state_dict(self.eval_network.state_dict())

        if train:
            return step, eps_reward, loss.detach().item()
        else:
            return step, eps_reward
        



def training_loop(env):

    total_episode = 1000
    dqn = DQN_Algorithm(env = env)
    episode_step_list = []
    episode_reward_list = []
    loss_list = []

    for eps in range(total_episode):

        
        if len(dqn.memory) < 100:
            episode_step, episode_reward = dqn.episode_generator()
            loss = None
        
        else:
            if eps % 10 == 0:
                episode_step, episode_reward, loss = dqn.episode_generator(copy_to_target = True, train = True)
            else:
                episode_step, episode_reward, loss = dqn.episode_generator(copy_to_target = False, train = True)
        
        print(f'\rEpisode {eps}, Step: {episode_step}, Average Steps: {mean(episode_step_list[-10::]):.2f}, Score: {episode_reward:.2f}, Average Score: {mean(episode_reward_list[-10::]):.2f}', end="")

        if eps % 100 == 0:
             print(f'\rEpisode {eps}, Step: {episode_step}, Average Steps: {mean(episode_step_list[-10::]):.2f}, Score: {episode_reward:.2f}, Average Score: {mean(episode_reward_list[-10::]):.2f}')


        episode_reward_list.append(episode_reward)
        episode_step_list.append(episode_step)
        loss_list.append(loss)

        


    plt.plot(episode_reward_list)
    #plt.plot(loss_list)
    plt.show()





if __name__ == '__main__':

    import World_Hardest_Game
    env = gym.make('CartPole-v0')
    env = gym.make('WorldHardestGame-v1')
    env = gym.make('LunarLander-v2')
    dqn = DQN_Algorithm(env)
    training_loop(env)
    
    
    
    """
    net = Network(4, 2)
    xxx = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    tmp_al = DQN_Algorithm(env, Network)
    print(net.forward(xxx))
    print(env.action_space.n)
    print(env.observation_space.shape)
    #print(DQN_Algorithm(env, Network))
    print(randint(10))
    print(torch.argmax(torch.Tensor([-1, 87, 999, -12])).item())
    print(tmp_al.choose_action(env.reset()))
    #print(net.state_dict())

    #print(torch.FloatTensor([[1, 2, 3], [4, 5, 6]]) + 1)
    print(random.sample(range(10), 3))
    

    memory = Replay_Memory()
    

    for _ in range(100):

        state = env.reset()
        action = randint(1)
        next_state, reward, done, _ = env.step(action)

        memory.push(state, action, reward, next_state)
        
    sampled_states, sampled_actions, sampled_rewards, sampled_next = memory.sample(3)
    xxx = net(sampled_states).detach()
    print(xxx)
    print(xxx.max(1)[0])
    print(sampled_rewards)
    print(xxx.max(1)[0] * 0.5 + sampled_rewards)
    
    """