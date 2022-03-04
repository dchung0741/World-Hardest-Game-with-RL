import gym
from gym import spaces
from World_Hardest_Game.envs.Game_1_pygame import Enemy, Game, WIDTH, HEIGHT
from numpy import array
from numpy.linalg import norm
from math import sqrt, exp, log10
from numpy.random import randint


class World_Hardest_Game_Env_1(gym.Env):

    def __init__(self) -> None:

        self.game = Game()
        self.action_space = spaces.Discrete(4)

        self.max_dist = sqrt(WIDTH**2 + HEIGHT**2)
        max_enemy_dist, max_food_dist, max_safe_dist = self.max_dist, self.max_dist, self.max_dist
        self.observation_space = spaces.Box(
            array([0, 0, 0, 0, 0, 0, 0, 0, 0]), 
            array([WIDTH, HEIGHT, max_enemy_dist, max_enemy_dist, max_enemy_dist, max_enemy_dist, max_food_dist, max_safe_dist, max_safe_dist]),
            dtype=float
        )


    def reset(self):

        self.game = Game()
        state = self.game.observe()
        closest_enemy_dist = min([state[2], state[3], state[4], state[5]])

        food_dist = state[6]
        safe_dist = state[7]
        goal_dist = state[8]

        self.cnt = 0
        self.safe_step = 0

        curr_pos = self.game.player.get_pos()
        self.player_traj = [ curr_pos ]

        self.closest_enemy_dist_list = [closest_enemy_dist]

        self.food_dist_list = [food_dist]
        self.goal_dist_list = [goal_dist]

        return self.game.observe()/100

    
    def step(self, action):

        reward = 0
        done = False
        self.cnt += 1


        # Event reward
        hit, eat, safe, goal = self.game.step(action)

        if hit:
            reward -= 100
            done = True

        if eat:
            reward += 10
        
        if safe:
            reward -= 1

        if goal:
            reward += 100
            done = True
        

        # Distance Reward
        curr_pos = self.game.player.get_pos()
        self.player_traj.append(curr_pos)
        last_pos = self.player_traj[-2]

        next_state = self.game.observe()

        closest_enemy_dist = min([next_state[2], next_state[3], next_state[4], next_state[5]])
        food_dist = next_state[6]
        safe_dist = next_state[7]
        goal_dist = next_state[8]
        
        self.closest_enemy_dist_list.append(closest_enemy_dist)
        self.food_dist_list.append(food_dist)
        self.goal_dist_list.append(goal_dist)

        
        if self.game.player.with_food <= self.game.all_food:
            #print('Food Mode')
            if self.food_dist_list[-1] >= self.food_dist_list[-2]:
                #print('Far from food')
                reward -= 5 #* exp(food_dist/100)
            else:
                #print('Closer to food')
                reward += 5 #* exp(food_dist/100)
            
            if self.goal_dist_list[-1] >= self.goal_dist_list[-2]:
                reward -= 0.05
            else:
                reward += 0.05
        
        
        else:
            #print('Goal Mode')
            if self.goal_dist_list[-1] > self.goal_dist_list[-2]:
                reward -= 0.1 * exp(goal_dist/100)
            else:
                reward += 0.1

        if closest_enemy_dist < 50:

            if self.closest_enemy_dist_list[-1] < self.closest_enemy_dist_list[-2]:
                #print('Proximity Alert')
                reward -= 1000 * closest_enemy_dist**-1
            else:
                reward += 0.5
        #print(f'Near by enemy punish: {10 * closest_enemy_dist**-1}')
        

        if curr_pos.all() != last_pos.all():
            reward += 1
        else:
            reward -= 1

        """
        if self.game.player.with_food < self.game.all_food:
            reward -= exp(food_dist/100)
        """
        
        if norm(curr_pos - last_pos) == 0:
            reward -= 50
        

        if self.cnt > 400:
            reward -= 10
            
        return next_state/100, reward, done, {}

    
    def render(self, close = False):
        self.game.view()

