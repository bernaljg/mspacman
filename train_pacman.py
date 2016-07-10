from pacman_agent import *
import gym
env = gym.make('MsPacman-v0')

p = Pacman_Agent(env)
p.learn(2,0.01)
