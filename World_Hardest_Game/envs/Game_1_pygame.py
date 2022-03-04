import pygame as pg
from numpy import array, concatenate
from numpy.random import randint
from numpy.linalg import norm

FPS = 60

WIDTH, HEIGHT = 1000, 500


# Color
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
LIGHT_GREEN = (0, 255, 128)

# Event
PLAYER_HIT = pg.USEREVENT + 1
PLAYER_WINS = pg.USEREVENT + 2


###########################################################################################################################################
class Player:

    def __init__(self, initial_x, initial_y) -> None:
        
        self.size = 20
        self.x = initial_x
        self.y = initial_y
        self.body = pg.Rect(self.x, self.y, self.size, self.size)
        self.velocity = 5
        self.with_food = 0
        #self.traj_memo = [array([self.x, self.y])]


    def get_pos(self):
        return array([self.x, self.y])
    
    def move(self, key_pressed):
        
        if key_pressed == 'up' and self.y - self.velocity >= 0:
            self.y -= self.velocity
            self.body = pg.Rect(self.x, self.y, self.size, self.size)
        
        if key_pressed == 'down' and self.y + self.velocity + self.size <= HEIGHT:
            self.y += self.velocity
            self.body = pg.Rect(self.x, self.y, self.size, self.size)
        
        if key_pressed == 'right' and self.x + self.velocity + self.size <= WIDTH:
            self.x += self.velocity
            self.body = pg.Rect(self.x, self.y, self.size, self.size)
        
        if key_pressed == 'left' and self.x - self.velocity >= 0:
            self.x -= self.velocity
            self.body = pg.Rect(self.x, self.y, self.size, self.size)
        
        #self.traj_memo.append(array([self.x, self.y]))


    def is_safe(self, safe_zones):

        safe = False
        goal = False
        
        for safe_zone in safe_zones:

            if safe_zone.x_boundaries[0] <= self.x <= safe_zone.x_boundaries[1] and safe_zone.y_boundaries[0] <= self.y <= safe_zone.y_boundaries[1]:
                
                if safe_zone.is_goal:
                    safe = True
                    goal = True
                    break
                else:
                    safe = True
                    break
        
        return safe, goal


    def pass_game(self, safe_zones, total_food):
        
        in_safe_zone, in_goal = self.is_safe(safe_zones)
        
        if in_safe_zone and self.with_food == total_food and in_goal:
            return True
        
        else:
            return False

###########################################################################################################################################
class Food:

    def __init__(self, x, y) -> None:

        self.size = 20
        self.x = x
        self.y = y
        self.body = pg.Rect(self.x, self.y, self.size, self.size)

    def get_pos(self):
        return array([self.x, self.y])
    
###########################################################################################################################################
class Enemy:

    def __init__(self, initial_x, initial_y, direction) -> None:
        
        self.size = 20
        self.x = initial_x
        self.y = initial_y
        self.body = pg.Rect(self.x, self.y, self.size, self.size)
        self.velocity = direction * 10
    
    def get_pos(self):
        return array([self.x, self.y])

    def move(self):
        
        if self.y + self.velocity >= HEIGHT or self.y + self.velocity  <= 0:
            self.velocity = - self.velocity

        self.y += self.velocity
        self.body = pg.Rect(self.x, self.y, self.size, self.size)



###########################################################################################################################################
class Safe_Zone:

    def __init__(self, top_left_x, is_goal) -> None:

        self.size = 20
        self.body = pg.Rect(top_left_x, 0, WIDTH//7, HEIGHT)
        self.x_boundaries = (top_left_x, top_left_x + WIDTH//7)
        self.y_boundaries = (0, HEIGHT)

        self.center = array([top_left_x + WIDTH//14, HEIGHT/2])

        self.is_goal = is_goal


###########################################################################################################################################
class Game:

    def __init__(self) -> None:

        pg.init()
        self.window = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption('World Hardest Game')

        # Game Characters        
        self.enemy_1 = Enemy(2 * WIDTH//7, HEIGHT//2, +1)
        self.enemy_2 = Enemy(3 * WIDTH//7, HEIGHT//2, -1)
        self.enemy_3 = Enemy(4 * WIDTH//7, HEIGHT//2, +1)
        self.enemy_4 = Enemy(5 * WIDTH//7, HEIGHT//2, -1)

        self.enemy_list = [self.enemy_1, self.enemy_2, self.enemy_3, self.enemy_4]

        self.food_list = [Food(3 * WIDTH//5, HEIGHT//4)]
        self.all_food = len(self.food_list)

        self.safe_1 = Safe_Zone(0, False)
        self.safe_2 = Safe_Zone(6 * WIDTH//7, True)
        self.safe_zone_list = [self.safe_1, self.safe_2]

        self.player = Player(*self.safe_1.center)


    def observe(self):
        
        
        cur_pos = array(self.player.get_pos())


        enemy_dist = []
        for enemy in self.enemy_list:
            enemy_pos = array(enemy.get_pos())
            dist = norm(enemy_pos - cur_pos)
            enemy_dist.append(dist)


        safe_dist = []
        for safe in self.safe_zone_list:
            dist = norm(cur_pos - safe.center)
            safe_dist.append(dist)
        

        food_dist = []
        if self.player.with_food == self.all_food:
            food_dist = [min(safe_dist)] * self.all_food

        else:
            
            for food in self.food_list:
                food_pos = array(food.get_pos())
                dist = norm(cur_pos - food_pos)
                food_dist.append(dist)
        

        return concatenate((cur_pos, enemy_dist, food_dist, safe_dist))

    
    ###########################################################################################################################################
    def check_state(self):

        hit = False
        for enemy in self.enemy_list:
            if self.player.body.colliderect(enemy.body):
                hit = True
                
        eat = False
        for food in self.food_list:
            if self.player.body.colliderect(food.body):
                
                self.food_list.remove(food)
                self.player.with_food += 1
                eat = True
        
        safe, _ = self.player.is_safe(self.safe_zone_list)
        
        goal = self.player.pass_game(self.safe_zone_list, self.all_food)
        

        return hit, eat, safe, goal        

    ###########################################################################################################################################
    def step(self, action):
    
        # Move the player
        if action == 0:
            self.player.move('up')
        
        elif action == 1:
            self.player.move('down')

        elif action == 2:
            self.player.move('right')
        
        elif action == 3:
            self.player.move('left')
        
        else:
            pass

        for enemy in self.enemy_list:
            enemy.move()

        # Check current state
        hit, eat, safe, goal = self.check_state()
        

        return hit, eat, safe, goal

    ###########################################################################################################################################
    def draw_window(self):

        self.window.fill(WHITE)

        pg.draw.rect(self.window, LIGHT_GREEN, self.safe_1.body)
        pg.draw.rect(self.window, LIGHT_GREEN, self.safe_2.body)
        
        pg.draw.rect(self.window, RED, self.player.body)

        for enemy in self.enemy_list:
            pg.draw.rect(self.window, BLUE, enemy.body)

        for food in self.food_list:
            pg.draw.rect(self.window, YELLOW, food.body)



    def view(self):

        clock = pg.time.Clock()        
        clock.tick(FPS)

        self.draw_window()
        pg.display.update()

        #pg.quit()
        
