import cv2
import numpy as np
import gym
from gym import spaces


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True):
        super().__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        
        if self.grayscale:
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(self.height, self.width, 1),
                dtype=np.uint8
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(self.height, self.width, 3),
                dtype=np.uint8
            )
    
    def observation(self, obs):
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            obs = np.expand_dims(obs, -1)
        return obs


class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
    
    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class CustomReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_health = 200
        self.prev_enemy_health = 200
        self.prev_rounds = 0
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_health = 200
        self.prev_enemy_health = 200
        self.prev_rounds = 0
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        health = info.get('health', 200)
        enemy_health = info.get('enemy_health', 200)
        rounds_won = info.get('rounds_won', 0)
        
        custom_reward = (
            1.0 * (health - self.prev_health) - 
            1.0 * (enemy_health - self.prev_enemy_health) + 
            10.0 * (rounds_won - self.prev_rounds)
        )
        
        self.prev_health = health
        self.prev_enemy_health = enemy_health
        self.prev_rounds = rounds_won
        
        return obs, custom_reward, done, info


def make_env(game_name='MortalKombatII-Genesis', state='Level1.LiuKangVsJax'):
    import retro
    env = retro.make(game=game_name, state=state)
    env = CustomReward(env)
    env = FrameSkip(env, skip=3)
    env = PreprocessFrame(env, width=84, height=84, grayscale=True)
    return env
