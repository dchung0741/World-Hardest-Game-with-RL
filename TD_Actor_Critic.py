from matplotlib import rc_params_from_file
import torch
from torch import FloatTensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gym
import matplotlib.pyplot as plt
from numpy import mean

########################################################################################################################
# Actor
########################################################################################################################
class Policy_Network(nn.Module):

    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x_in):
        return self.net(x_in)



class Actor:

    def __init__(self, env, entropy_param = 0) -> None:
        
        self.action_dim = env.action_space.n
        self.observation_dim = env.observation_space.shape[0]

        self.net = Policy_Network(self.observation_dim, self.action_dim)
        
        self.optim = optim.Adam(self.net.parameters(), lr = 1e-4)

        self.actor_loss = []
        self.entropy_param = entropy_param
    

    def update_entropy_param(self, disc):
        self.entropy_param *= disc


    def choose_action(self, state):
        
        state_tensor = FloatTensor(state)
        net_output = self.net(state_tensor).softmax(-1)
        
        #print(net_output.detach())
        dist = Categorical(net_output)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        #print(entropy)

        return action.item(), log_prob, entropy


    def learn(self, td_error, log_prob, entropy):
        
        loss = - td_error * log_prob - self.entropy_param * entropy

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.actor_loss.append(loss.item())


########################################################################################################################
# Critic
########################################################################################################################
class Value_Network(nn.Module):

    def __init__(self, in_dim) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x_in):
        return self.net(x_in)


class Critic:

    def __init__(self, env) -> None:

        self.observation_dim = env.observation_space.shape[0]
        self.net = Value_Network(self.observation_dim)

        self.optim = optim.Adam(self.net.parameters(), lr = 1e-4)
        self.loss = nn.MSELoss()
        self.critic_loss = []

    def td_error(self, state, reward, next_state, done, gamma = 0.99, train = False):
        
        state_tensor = FloatTensor(state)
        next_state_tensor = FloatTensor(next_state)
        reward = torch.tensor(reward, dtype=torch.float)
        value = self.net(state_tensor)
        next_value = self.net(next_state_tensor)

        self.optim.zero_grad()
        td_error = reward + gamma * next_value * (1 - int(done)) - value
        self.critic_loss.append((td_error**2).item())

        if train:
            loss = (reward + gamma * next_value * (1 - int(done))- value)**2
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        return td_error.item()    


########################################################################################################################
# Actor - Critic
########################################################################################################################

class A2C:

    def __init__(self, env, entropy_param = 0, reward_scale = 1) -> None:
        
        self.env = env
        self.actor = Actor(env, entropy_param)
        self.critic = Critic(env)
        self.reward_scale = reward_scale
        

    def train(self):
        
        steps = 0
        eps_reward = 0
        state = self.env.reset()

        for _ in range(self.env.spec.max_episode_steps):

            steps += 1

            action, log_prob, entropy = self.actor.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)
            
            eps_reward += reward
            """
            if str(self.env) == '<TimeLimit<CartPoleEnv<CartPole-v0>>>':
                x, _, theta, _ = next_state
                r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8 
                r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                reward = (r1 + r2) * 10
            """
            td_error = self.critic.td_error(state, reward * self.reward_scale, next_state, done, train = True)
            self.actor.learn(td_error, log_prob, entropy)    

            if done:
                break
            else:
                state = next_state
                self.env.render()
        
        return steps, eps_reward


def training_loop(env):

    a2c = A2C(env, entropy_param = 0e-1, reward_scale = 1e-3)

    total_episode = 300
    episode_steps = []
    episode_reward = []
    
    for eps in range(total_episode):

        eps_step, eps_reward = a2c.train()
        episode_steps.append(eps_step)
        episode_reward.append(eps_reward)

        print(f'\rEpisode {eps},  Average Steps: {mean(episode_steps[-10::]):.2f}, Average Score: {mean(episode_reward[-10::]):.2f}', end="")

        if eps % 100 == 0:
             print(f'\rEpisode {eps},  Average Steps: {mean(episode_steps[-10::]):.2f}, Average Score: {mean(episode_reward[-10::]):.2f}')

        a2c.actor.update_entropy_param(1)
    
    #plt.plot(a2c.actor.actor_loss[::200])
    #plt.plot(a2c.critic.critic_loss[::200])
    #plt.show()
    
    step_moving_avg = [episode_steps[:i][-10:] for i in range(len(episode_steps))]
    step_moving_avg = list(map(mean, step_moving_avg))
    plt.plot(episode_steps)
    plt.plot(step_moving_avg)
    plt.show()
    


if __name__ == '__main__':
    
    import World_Hardest_Game
    from numpy.random import randint

    env = gym.make('WorldHardestGame-v1')
    #env = gym.make('CartPole-v0')
    #env = gym.make('LunarLander-v2')
    training_loop(env)
    
    """
    state = env.reset()
    print(state)
    for _ in range(200):
        env.game.step(randint(3))
        env.render()
    """

    state_0 = [ 71.42857143, 250., 213.57142857, 356.57142857, 499.57142857, 642.57142857, 428.57142857, 0., 857.]
    state_0 = FloatTensor(state_0)
    
    actor = Actor(env)
    
    state = env.reset()
    
    print(type(state))
    print(actor.net(FloatTensor(state)))
    print(actor.net(FloatTensor(state)).softmax(-1))

    """
    actions = []

    for _ in range(200):
        action, log_prob = actor.choose_action(state)
        actions.append(action)
        new_state, reward, done, _ = env.step(action)
        state = new_state
        env.render()
        if done:
            state = env.reset()


    print(actions)
    """