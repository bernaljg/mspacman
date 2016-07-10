
# coding: utf-8

# In[3]:

import tensorflow as tf
import gym


# In[5]:

"""Testing Document

Fully connected two layer network to perform Q-learning for pacman

The basic relevant features for the game state are:
-MsPacman's location: 2 values
-Four pairs of distances from each ghost: 8 values
    
Total of 10 input values.
*If there are less ghosts we can set an arbitrarily high number for their distances.
The hope is that the map will be encoded automatically by the neural network structure.

The network will compute Q values for each action "a" from a state "s". We will then choose an action. 
"""

def model():
    env = gym.make('MsPacman-v0')

    game_state = tf.placeholder("float32",[10])

    w1_init = tf.random_normal((100,10), name = "w1_init")
    w2_init = tf.random_normal((9,100), name = "w2_init")

    w1 = tf.Variable(w1_init, name="w1")
    w2 = tf.Variable(w2_init, name="w2")

    y1 = tf.matmul(w1,state)
    layer1_act = tf.nn.relu(y1)

    y2 = tf.matmul(w2,layer1_act)
    Q_vals = tf.nn.log_softmax(y2)

    action = tf.arg_max(Q_vals)

    obs,_,reward,done = env.step(action)

    new_state = extract_features(obs)

    new_y1 = tf.matmul(w1,new_state)
    new_layer1_act = tf.nn.relu(new_y1)

    new_y2 = tf.matmul(w2,new_layer1_act)
    new_Q_vals = tf.nn.log_softmax(new_y2)
    
    max_Q = tf.maximum(new_Q_vals)    
    
    new_Q_vals = np.copy(Q_vals)
    new_Q_Vals[action] = reward + max_Q
    
    loss = tf.nn.l2_loss((new_Q_vals - Q_vals))
    
    tf.train.GradientDescentOptimizer.minimize(loss)




# In[6]:

loss


# In[ ]:



