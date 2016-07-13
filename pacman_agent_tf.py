"""Created by: Bernal Jimenez
06/30/2016
This program contains an agent that uses Deep Q-Learning to play the OpenAI Gym version of MsPacman."""

import gym
import numpy as np
import tensorflow as tf
from character_extraction import *
import pickle

class DRL_Model(object):

    def __init__(self, env):
        self.env = env
        self.g = tf.Graph()

        with self.g.as_default():
            w1 = tf.Variable(np.random.random((30,10)), name="w1")
            w2 = tf.Variable(np.random.random((9,30)), name="w2")
            lr = tf.Variable(tf.constant(0.01), name="lr")
            init_op = tf.initialize_all_variables()

            game_state = extract_features(observation)
            game_state = np.reshape(game_state, (10,1))

            state = tf.placeholder("float64",[10,1])
            new_state = tf.placeholder("float64",[10,1])
            
            y1 = tf.matmul(w1,state)
            layer1_act = tf.nn.relu(y1)

            y2 = tf.matmul(w2,layer1_act)
            Q_vals = tf.nn.log_softmax(y2)
        
            #Epsilon Greedy
            if np.random.randn(1) <= eps:
                action = self.env.action_space.sample()
            else:
                action = np.argmax(Q_values)

            #Perform Action
            game_info = self.env.step(action)
            new_observation,reward,done,_ = game_info

            new_state = extract_features(new_observation)
            new_state = np.reshape(game_state, (10,1))
            
            n_y1 = tf.matmul(w1,state)
            n_layer1_act = tf.nn.relu(n_y1)

            n_y2 = tf.matmul(w2,layer1_act)
            Q2_vals = tf.nn.log_softmax(n_y2)
            
            max_Q = tf.maximum(Q2_vals,0)            
            new_Q_vals = np.copy(Q_vals)
            new_Q_vals[action] = reward + lr*max_Q

            loss = tf.nn.l2_loss((new_Q_vals - Q_vals))
            train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
        
    def forward_pass(self, observation):
        """This method makes a forward pass through the model to obtain the predicted Q values for different actions from the current observation"""
        return None
        
    def find_best_action(self, observation):
        Q_values = self.forward_pass(observation)
        action = np.argmax(Q_values)
        return action

    def update_weights(self, observation, eps):
        """This method updates the weights through backpropagation from a Q-Learning defined loss"""
        with tf.Session(graph = self.g) as sess:
            sess.run(init_op)
            _,loss,new_observation = sess.run([train_step, loss, new_observation], feed_dict={state: game_state})
       
        return new_observation

class Pacman_Agent(object):

    def __init__(self, env):
        self.env = env
        self.model = None

    def act(self, observation):
        """This method extracts features from pixel space and runs the policy to determine the next action.""" 
        if self.model == None:
            print("Learn First: Method self.learn()")
        action = self.model.find_best_action(observation)
        return action

    def learn(self, num_epochs, eps):
        """This method updates the model object until the agent is a true MsPacman Playa"""
        done = False
        self.model = DRL_Model(self.env)
        for i in range(num_epochs):
            observation = self.env.reset()
            while not(done):
                observation = self.model.update_weights(observation, eps)
            pickle.dump(self.model,open("trained_model.p","wb"))

        print("Done with learning")

    def play(self):
        done = False
        obs = self.env.reset()
        while not(done): 
            action = self.act(obs)
            obs,_,done,_ = self.env.step(action)
            self.env.render()
