# Playing the World Hardest Game with Reinforcement Learning

This is an attempt to solve the world hardest game with several RL algorithms.
If you do not know about the game, please check: https://www.youtube.com/watch?v=C14VDpGAbSE&t=382s&ab_channel=CodeBullet (This video is not made by me.)


## My game
I create a simple version of the game with pygame.

- The game that can be played by human is World_Hardest_Game/envs/Game_1_pygame_human.py
- The game that can be played by AI is World_Hardest_Game/envs/Game_1_pygame.py
- The reward function is in World_Hardest_Game/envs/custom_env.py

## Best performance

- I try to use the A2C algorithm to solve the game. Video: https://youtu.be/w1T6CeUWNh8 (This is my training result). I saved the model in the "WorldHardestGame_A2C_Model" folder.
- DQN has the best performace so far within 200 episodes, but further training gives worse results.

