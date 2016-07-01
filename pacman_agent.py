"""Created by: Bernal Jimenez
06/30/2016
This program contains an agent that uses Deep Q-Learning to play the OpenAI Gym version of MsPacman."""

import gym
import numpy as np
import tensorflow as tf

def extract_features(frame):    
"""This function takes in a frame and extracts useful features to create a more manageable game state space"""

class DRL_Model(object):

    def __init__(self):
    
    def find_best_action(self, observation):
    """This method makes a forward pass through the model to obtain the predicted Q values for different actions from the current observation"""

    def update_weights(self):
    """This method updates the weights through backpropagation from a Q-Learning defined loss"""

class MsPacman_Playa2000(object):

    def __init__(self, observation, action_space, model)
        self.observation = 
        self.action_space = action_space
        self.model = model

    def act(self, observation):
    """This method extracts features from pixel space and runs the policy to determine the next action.""" 
        
        action = self.model.find_best_action(game_state)
        return action

    def learn(self, env, num_episodes, eps):
    """This method updates the model object until the agent is a true MsPacman Playa"""
        for i in range(num_episodes):
            if np.random.random <= eps:
               self.action.sample() 
