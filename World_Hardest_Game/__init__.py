from gym.envs.registration import register

register(
    id = 'WorldHardestGame-v1',
    entry_point = 'World_Hardest_Game.envs:World_Hardest_Game_Env_1',
    max_episode_steps = 500,
)