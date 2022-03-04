import pygame as pg
from numpy.random import randint

FPS = 60

WIDTH, HEIGHT = 1000, 500
WIN = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption('World Hardest Game')


# Color
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
LIGHT_GREEN = (0, 255, 128)


# Action
UP = pg.K_UP
DOWN = pg.K_DOWN
RIGHT = pg.K_RIGHT
LEFT = pg.K_LEFT

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

    
    def move(self, key_pressed):
        
        if key_pressed[UP] and self.y - self.velocity >= 0:
            self.y -= self.velocity
            self.body = pg.Rect(self.x, self.y, self.size, self.size)
        
        if key_pressed[DOWN] and self.y + self.velocity + self.size <= HEIGHT:
            self.y += self.velocity
            self.body = pg.Rect(self.x, self.y, self.size, self.size)
        
        if key_pressed[RIGHT] and self.x + self.velocity + self.size <= WIDTH:
            self.x += self.velocity
            self.body = pg.Rect(self.x, self.y, self.size, self.size)
        
        if key_pressed[LEFT] and self.x - self.velocity >= 0:
            self.x -= self.velocity
            self.body = pg.Rect(self.x, self.y, self.size, self.size)


    def pass_game(self, safe_zones, total_food):

        in_safe_zone = False

        for safe in safe_zones:

            if safe.x_boundaries[0] <= self.x <= safe.x_boundaries[1] and safe.y_boundaries[0] <= self.y <= safe.y_boundaries[1]:
                in_safe_zone = True
                break

        if in_safe_zone and self.with_food == total_food:
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
        
    
###########################################################################################################################################
class Enemy:

    def __init__(self, initial_x, initial_y, direction) -> None:
        
        self.size = 20
        self.x = initial_x
        self.y = initial_y
        self.body = pg.Rect(self.x, self.y, self.size, self.size)
        self.velocity = direction * 10
    
    def move(self):
        
        if self.y + self.velocity >= HEIGHT or self.y + self.velocity  <= 0:
            self.velocity = - self.velocity

        self.y += self.velocity
        self.body = pg.Rect(self.x, self.y, self.size, self.size)



###########################################################################################################################################
class Safe_Zone:

    def __init__(self, top_left_x) -> None:

        self.size = 20
        self.body = pg.Rect(top_left_x, 0, WIDTH//7, HEIGHT)
        self.x_boundaries = (top_left_x, top_left_x + WIDTH//7)
        self.y_boundaries = (0, HEIGHT)

###########################################################################################################################################
# Game Characters
player = Player(randint(WIDTH//7), randint(HEIGHT))

enemy_1 = Enemy(2 * WIDTH//7, HEIGHT//2, +1)
enemy_2 = Enemy(3 * WIDTH//7, HEIGHT//2, -1)
enemy_3 = Enemy(4 * WIDTH//7, HEIGHT//2, +1)
enemy_4 = Enemy(5 * WIDTH//7, HEIGHT//2, -1)
enemies = [enemy_1, enemy_2, enemy_3, enemy_4]

food_list = [Food(WIDTH//2, HEIGHT//2)]
all_food = len(food_list)

safe_1 = Safe_Zone(0)
safe_2 = Safe_Zone(6 * WIDTH//7)

###########################################################################################################################################
def handle_collison(player: Player, enemy_list: list, food_list: list):

    for enemy in enemy_list:
        if player.body.colliderect(enemy.body):
            pg.event.post(pg.event.Event(PLAYER_HIT))

    
    for food in food_list:
        if player.body.colliderect(food.body):
            
            food_list.remove(food)
            player.with_food += 1
    
    if player.pass_game([safe_1, safe_2], all_food):
        pg.event.post(pg.event.Event(PLAYER_WINS))


###########################################################################################################################################
def draw_window():

    WIN.fill(WHITE)
    pg.draw.rect(WIN, LIGHT_GREEN, safe_1.body)
    pg.draw.rect(WIN, LIGHT_GREEN, safe_2.body)
    
    pg.draw.rect(WIN, RED, player.body)

    for enemy in enemies:
        pg.draw.rect(WIN, BLUE, enemy.body)

    for food in food_list:
        pg.draw.rect(WIN, YELLOW, food.body)



def main():

    
    clock = pg.time.Clock()
    run = True

    while run:
        
        clock.tick(FPS)

        for event in pg.event.get():

            # Check quit
            if event.type == pg.QUIT:
                run = False

            if event.type == PLAYER_HIT:
                run = False
            
            if event.type == PLAYER_WINS:
                run = False
        

        key_pressed = pg.key.get_pressed()
        player.move(key_pressed = key_pressed)
        for enemy in enemies:
            enemy.move()
        
        handle_collison(player, enemies, food_list)

        draw_window()
        pg.display.update()

    pg.quit()
    
pg.init()
if __name__ == '__main__':
    main()
    