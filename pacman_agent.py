"""Created by: Bernal Jimenez
06/30/2016
This program contains an agent that uses Deep Q-Learning to play the OpenAI Gym version of MsPacman."""

import gym
import numpy as np
from character_extraction import *
import pickle
import os

class DRL_Model(object):

    def __init__(self, env):
        self.env = env
        self.w1 = np.random.random((30,10))
        self.w2 = np.random.random((9,30))
        self.lr = 0.0001
        
    def forward_pass(self, observation):
        """This method makes a forward pass through the model to obtain the predicted Q values for different actions from the current observation"""
        game_state = extract_features(observation)

        y1 = self.w1.dot(game_state)
        layer1_act = 1./(1.+np.exp(-y1))

        y2 = self.w2.dot(layer1_act)
        Q_vals = np.exp(y2)/np.sum(np.exp(y2))
    
        return Q_vals, y1, y2, layer1_act, game_state

    def update_weights(self, observation, eps, previous_step = None):
        """This method updates the weights through backpropagation from a Q-Learning defined loss"""
        if previous_step == None:
            Q_vals, y1, y2, layer1_act, state = self.forward_pass(observation)
        else:
            Q_vals, y1, y2, layer1_act, state = previous_step

        action = np.argmax(Q_vals)
        
        #Epsilon Greedy
        if np.random.randn(1) <= eps:
            action = self.env.action_space.sample()

        #Perform Action
        game_info = self.env.step(action)
        new_observation,reward,done,_ = game_info
        
        self.reward += reward/1000.

        new_Q_vals,new_y1,new_y2,new_layer1_act,new_game_state = self.forward_pass(new_observation)
        
        max_Q = np.max(new_Q_vals)            
        predicted_Q_vals = np.copy(Q_vals)
        predicted_Q_vals[action] = self.reward + self.lr*max_Q
        
        diff = Q_vals - predicted_Q_vals
        self.loss = np.sum(np.square(diff))
        
        #W2 Computation
        dL_dQ = diff
        
        #Computing Softmax Derivative
        dQ_dy2 = -1/2.*np.outer(Q_vals, Q_vals)
        dQ_dy2 = dQ_dy2 - np.diag(np.diag(dQ_dy2))
        dQ_dy2 += np.diag(Q_vals*(1-Q_vals))
        dQ_dy2 = np.sum(dQ_dy2, 0)
        
        
        dL_dy2 = dL_dQ*dQ_dy2
        dy2_dw2 = layer1_act
        
        """Matrix W2 Update Expression"""
        dL_dw2 = np.outer(dL_dy2, dy2_dw2)

        #W1 Computation
        dy2_dlayer1_act = self.w2
        dL_dlayer1_act = dy2_dlayer1_act.T.dot(dL_dy2)

        dlayer1_act_dy1 = layer1_act*(1-layer1_act)
        
        dL_dy1 = dL_dlayer1_act*dlayer1_act_dy1
        
        """W1 Update Expression"""
        dL_dw1 = np.outer(dL_dy1, state)
        
        self.w1 -= dL_dw1
        self.w2 -= dL_dw2
        
        next_step = (new_Q_vals, new_y1, new_y2, new_layer1_act, new_game_state)
        return next_step, done

class Pacman_Agent(object):

    def __init__(self):
        env = gym.make('MsPacman-v0')
        self.env = env
        self.model = None

    def act(self, observation):
        """This method extracts features from pixel space and runs the policy to determine the next action.""" 
        Q_vals,_,_,_,_ = self.model.forward_pass(observation)
        action = np.argmax(Q_vals)
        return action

    def train(self, num_epochs, eps):
        """This method updates the model object until the agent is a true MsPacman Playa"""
        self.model = DRL_Model(self.env)
        for i in range(num_epochs):
            if i%250 == 0:
                eps = eps/1.5
            done = False
            observation = self.env.reset()
            next_step = None
            self.model.reward = 0
            while not(done):
                next_step, done = self.model.update_weights(observation, eps, next_step)
                print(self.model.loss)
            pickle.dump(self.model,open("trained_model.p","wb"))
            print("Done with {} epochs".format(i))

    def play(self):
        if os.path.exists("trained_model.p"):
            self.model = pickle.load(open("trained_model.p","rb"))
        else:
            print("Train model first using self.train()")
            return
        done = False
        obs = self.env.reset()
        while not(done): 
            action = self.act(obs)
            obs,_,done,_ = self.env.step(action)
            self.env.render()
