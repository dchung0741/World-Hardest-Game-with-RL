import torch
from torch import FloatTensor, LongTensor
import  torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
import random
import gym
from numpy import array, convolve, ones
import matplotlib.pyplot as plt

##############################################################################################################
# Replay Buffer
##############################################################################################################
class Replay_Memory:

    def __init__(self, capacity) -> None:
        
        self.memory = []
        self.capacity = capacity
        self.log_num = 0
    

    def push(self, state, action, reward, next_state, is_done):

        memo = [state, action, reward, next_state, is_done]

        if len(self) < self.capacity:
            self.memory.append(memo)

        else:
            idx = self.log_num % self.capacity
            self.memory[idx] = memo

        self.log_num += 1
    

    def __len__(self):
        return len(self.memory)


    def sample(self, batch_size):

        sampled_idx = random.sample(range(len(self)), batch_size)
        
        
        sampled_state = []
        sampled_action = []
        sampled_reward = []
        sampled_next_state = []
        sampled_is_done = []

        for idx in sampled_idx:
            
            sampled_state.append(self.memory[idx][0])
            sampled_action.append(self.memory[idx][1])
            sampled_reward.append(self.memory[idx][2])
            sampled_next_state.append(self.memory[idx][3])
            sampled_is_done.append(int(self.memory[idx][4]))
        

        sampled_state = FloatTensor(array(sampled_state))
        sampled_action = LongTensor(array([sampled_action])).reshape(batch_size, 1)
        sampled_reward = FloatTensor(array([sampled_reward])).reshape(batch_size, 1)
        sampled_next_state = FloatTensor(array(sampled_next_state))
        sampled_is_done = LongTensor(array([sampled_is_done])).reshape(batch_size, 1)

        
        return sampled_state, sampled_action, sampled_reward, sampled_next_state, sampled_is_done





    
##############################################################################################################
# Policy Network
##############################################################################################################
class Policy_Network(nn.Module):

    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    
    def forward(self, state):

        return self.network(state)




##############################################################################################################
# Q Network
##############################################################################################################
class Q_Network(nn.Module):

    def __init__(self, obs_dim, act_dim) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim + act_dim - 1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state, action):
        
        state_action = torch.cat([state, action], dim = 1)

        return self.network(state_action)





##############################################################################################################
# SAC Algorithm
##############################################################################################################
class SAC_Algorithm:

    def __init__(self, env, lr = 1e-4, alpha = 0.2, gamma = 0.99, rho = 0.005, batch_size = 100) -> None:


        # Environments
        self.env = env
        self.observ_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        #self.action_dim = env.action_space.shape[0]

        # Reply Memory
        self.replay = Replay_Memory(10000)

        # Networks
        # NOTE: 
        # This code is implementing the SAC psuedocode given in https://spinningup.openai.com/en/latest/algorithms/sac.html#pseudocode,
        # where there is no value network to be trained. 

        self.policy_network = Policy_Network(self.observ_dim, self.action_dim)
        self.policy_optim = Adam(self.policy_network.parameters(), lr = lr)

        self.Q_net_1 = Q_Network(self.observ_dim, self.action_dim)
        self.Q1_optim = Adam(self.Q_net_1.parameters(), lr = lr)
        self.Q1_loss = nn.MSELoss()

        self.Q_net_2 = Q_Network(self.observ_dim, self.action_dim)
        self.Q2_optim = Adam(self.Q_net_2.parameters(), lr = lr)
        self.Q2_loss = nn.MSELoss()
        
        self.Q_tar_1 = Q_Network(self.observ_dim, self.action_dim)
        self.Q_tar_1.load_state_dict(self.Q_net_1.state_dict())

        self.Q_tar_2 = Q_Network(self.observ_dim, self.action_dim)
        self.Q_tar_2.load_state_dict(self.Q_net_2.state_dict())


        # Parameters
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.batch_size = batch_size


        self.total_step_cnt = 0
        self.update_cnt = 50
        self.actor_loss_history = []
        self.critic_loss_history = []




    def choose_Action(self, state, batch_mode = False):

        state = FloatTensor(state)
        policy_network_out = self.policy_network(state)

        if batch_mode is False:

            cata = Categorical(policy_network_out.softmax(0))
            action = cata.sample(torch.Size((1, )))
            log_prob = cata.log_prob(action)

            return action.item(), log_prob

        else:
            
            cata = Categorical(policy_network_out.softmax(1))
            action = cata.sample(torch.Size((1, )))
            log_prob = cata.log_prob(action)
            col = state.shape[0]

            return action.reshape(col, 1), log_prob.reshape(col, 1)


    def episode_Generator(self, max_step = 200, mode = 'collect'):
        
        state = self.env.reset()
        eps_step_cnt = 0

        for _ in range(max_step):
            
            action, log_prob = self.choose_Action(state)
            next_state, reward, done, _ = self.env.step(action)
            eps_step_cnt += 1

            

            # Save to replay memory
            self.replay.push(state, action, reward, next_state, done)

            if mode == 'train':
                self.total_step_cnt += 1

                if (self.total_step_cnt // self.update_cnt) % 2 == 0:
                    #if (self.total_step_cnt // self.update_cnt) == (self.total_step_cnt / self.update_cnt):
                    if (self.total_step_cnt % self.update_cnt) == 0:
                        print('Updating Network')

                    actor_loss, critic_loss = self.learn()

                    self.actor_loss_history.append(actor_loss)
                    self.critic_loss_history.append(critic_loss)


            if done:
                break

            else:
                state = next_state
                self.env.render()
                
        return eps_step_cnt
    

    def learn(self):

        # Sample from replay memory
        replay_size = len(self.replay)
        batch_size = min(self.batch_size, replay_size // 10)
        sampled_state, sampled_action, sampled_reward, sampled_next_state, sampled_is_done = self.replay.sample(batch_size)

        
        # 1. Compute target y(r, s', done) = r + gamma * (1 - done) * (min Q_{target, i}(s', a') - alpha * log pi(a'| s'))

        # 1) Sample action for state s' from policy network
        next_action, next_action_log_prob = self.choose_Action(sampled_next_state, batch_mode = True)

        # 2) Q values of (s', a')from target network
        Q_tar_val_1, Q_tar_val_2 = self.Q_tar_1.forward(sampled_next_state, next_action), self.Q_tar_2.forward(sampled_next_state, next_action)
        Q_min = Q_tar_val_1.minimum(Q_tar_val_2)
        Q_min = Q_min.detach()

        # 3) Target y
        y = sampled_reward + self.gamma * (1 - sampled_is_done) * (Q_min - self.alpha * next_action_log_prob.detach())

        
        # 2. Update Q_i
        Q1_val, Q2_val = self.Q_net_1.forward(sampled_state, sampled_action), self.Q_net_2.forward(sampled_state, sampled_action)
        Q1_loss_val, Q2_loss_val = self.Q1_loss(Q1_val, y)/batch_size, self.Q2_loss(Q2_val, y)/batch_size
        
        self.Q1_optim.zero_grad()
        Q1_loss_val.backward()
        self.Q1_optim.step()

        self.Q2_optim.zero_grad()
        Q2_loss_val.backward()
        self.Q2_optim.step()
        
        
        # 3. Update Policy network

        # 1) Sample action for state s from policy network
        this_action, this_action_log_prob = self.choose_Action(sampled_state, batch_mode = True)

        # 2) Compute Q_target(s, a), where a is sampled from 'this_action'
        Q_net_val_1, Q_net_val_2 = self.Q_net_1.forward(sampled_state, this_action), self.Q_net_2.forward(sampled_state, this_action)
        Q_min = Q_net_val_1.minimum(Q_net_val_2)
        Q_min = Q_min.detach()
        
        # 3) Compute loss
        # SIGN????????????????????????????????????????????
        policy_loss_val = - (Q_min - self.alpha * this_action_log_prob).sum() / batch_size
        
        self.policy_optim.zero_grad()
        policy_loss_val.backward()
        self.policy_optim.step()


        # 4. Copy Q_i weights to Q_targets
        tar_net_param_pair = [(self.Q_tar_1.parameters(), self.Q_net_1.parameters()), 
                                (self.Q_tar_2.parameters(), self.Q_net_2.parameters())]
        for pair in tar_net_param_pair:
            for target_param, param in zip(*pair):
                target_param.data.copy_(target_param * (1 - self.rho) + param * self.rho)
            
        
        return policy_loss_val, Q1_loss_val + Q2_loss_val


##############################################################################################################
# Traininig Loop
##############################################################################################################
def moving_avg(data_array, width = 10):
    
    data_array = array(data_array)
    filter = ones(width)
    return convolve(data_array, filter, 'valid') / width


def traininig_loop(env):
    
    sac = SAC_Algorithm(env, lr = 1e-3, batch_size = 100)
    total_episode = 1000
    collect_eps = total_episode // 10
    collect_cnt = 0

    step_list = []

    for eps in range(total_episode):

        
        
        if eps < collect_eps:
            step = sac.episode_Generator(mode = 'collect')
        
        else:
            step = sac.episode_Generator(mode = 'train')
        
        print('\r', f'Episode: {eps}, Step: {step}', end = ' ')
        step_list.append(step)
        collect_cnt += 1


    fig, (ax1, ax2, ax3) = plt.subplots(3,)
    ax1.plot(moving_avg(step_list), 'r')
    ax2.plot(moving_avg(torch.stack(sac.actor_loss_history).detach().numpy()), 'g')
    ax3.plot(moving_avg(torch.stack(sac.critic_loss_history).detach().numpy()), 'b')
    plt.show()




if __name__ == '__main__':

    #env = gym.make('LunarLander-v2')
    #traininig_loop(env)
    #env.close()
    

    

    """
    env = gym.make('Pendulum-v0')
    #traininig_loop(env)
    state = env.reset()
    done = False
    while done is False:
        action = random.random()
        _, _, done, _, = env.step([action])
        env.render()

    env.render()
    env.close()
    """
    
    """
    env = gym.make('CartPole-v0')
    act_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    state = env.reset()
    print(act_dim)
    print(obs_dim)

    print('----------------------------------------------------------------------------------')
    print('Test Policy Network')
    print('----------------------------------------------------------------------------------')

    pn_test = Policy_Network(obs_dim, act_dim)
    state = FloatTensor(state)
    prob = pn_test.forward(state).softmax(0)
    cata = Categorical(prob)
    print(cata.sample(torch.Size((1, ))).item())
    print(torch.log(prob))
    print(cata.log_prob(torch.tensor([0])))
    print(cata.log_prob(torch.tensor([1])))


    print('----------------------------------------------------------------------------------')
    print('Test Replay & policy & Q & SAC')
    print('----------------------------------------------------------------------------------')
    state = env.reset()
    done = False
    sac = SAC_Algorithm(env)

    for _ in range(200):
        
        action, _ = sac.choose_Action(state)
        next_state, reward, done, _ = env.step(action)
        sac.replay.push(state, action, reward, next_state, done)

        if done:
            break
        else:
            state = next_state
        
    print(f'memo len = {len(sac.replay)}')
    state, action, reward, next_state, is_done = sac.replay.sample(3)
    print('Sampled output')
    print(f'state: {state}')
    print(f'action: {action}')
    print(f'reward: {reward}')
    print(f'done: {is_done}')
    print(f'1 - done: {1 - is_done}')
    print(torch.cat([state, action], dim = 1))

    print('Q_output')
    Q_test = Q_Network(obs_dim, act_dim)
    print(Q_test.forward(state, action))

    print('Policy_output')
    act_test, prob_test = sac.choose_Action(state, batch_mode = True)
    print(act_test)
    print(prob_test)
    print(sac.policy_network.forward(state))
    print(sac.policy_network.forward(state).softmax(1))

    print('Q_target_output')
    Q_tar_1_test = sac.Q_tar_1.forward(state, action)
    Q_tar_2_test = sac.Q_tar_2.forward(state, action)
    Q_cat_test = torch.cat([Q_tar_1_test, Q_tar_2_test], dim = 1)
    #print(Q_cat_test.minimum(dim = 1))
    print(Q_tar_1_test)
    print(Q_tar_2_test)
    print(Q_tar_1_test.minimum(Q_tar_2_test))
    print(Q_tar_1_test.min(Q_tar_2_test))

    print('----------------------------------------------------------------------------------')
    print('Test SAC - training')
    print('----------------------------------------------------------------------------------')
    
    sac_test = SAC_Algorithm(env)
    for _ in range(10):
        sac_test.episode_Generator()
    print(len(sac_test.replay))
    print(sac_test.episode_Generator(mode = 'train'))
    
    print('----------------------------------------------------------------------------------')
    print('Play Ground')
    print('----------------------------------------------------------------------------------')
    
    xxx = FloatTensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    yyy = LongTensor([[6], [0]])
    print(torch.cat([xxx, yyy], dim = 1))
    print(xxx.sum())

    """

    memory = Replay_Memory(3)
    memory.push(1, 1, 1, 1, 1)
    print(memory.memory)
    memory.push(2, 2, 2, 2, 2)
    print(memory.memory)
    memory.push(3, 3, 3, 3, 3)
    print(memory.memory)
    memory.push(4, 4, 4, 4, 43)
    print(memory.memory)