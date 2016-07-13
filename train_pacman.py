from pacman_agent import *
import gym
env = gym.make('MsPacman-v0')

p = Pacman_Agent(env)
p.learn(1000,0.9)
