"""
This notebook will be used to test the MsPacman OpenAI Gym environment for feature extraction.
"""

import tensorflow as tf 
import gym 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp 
from sklearn import preprocessing 
import scipy.ndimage 
import pickle 
 
def load_char_pickle():
    """Loads character templates"""
    return pickle.load(open("characters.p","rb"),encoding='latin1') 
 
def normalize_image(image): 
    """Scales images with RGB Channels
    
    args: Image
    returns: Normalized image with respect to all its axes
    """ 
    image = image.astype('float32') 
    image = image - np.mean(image)
    image = image/np.std(image)
     
    return image 
 
def find_characters(raw_frame, pacman_track): 
    """Finds character locations by using convolution with templates
    args: 
        Raw Frame Data
        Character Template Dictionary
        Pacman Starting Point
        
    returns:
        Dictionary with character locations
    """
    #Preprocessing Data 
    raw_frame = raw_frame[:172]
    raw_frame = normalize_image(raw_frame) 
    
    #for i,image in enumerate(characters): 
    #    characters[i] = normalize_image(image)
    characters = load_char_pickle()
    
    #Formatting Game frame to match TensorFlow Convolution
    raw_frame = np.array([raw_frame])

    character_filters = np.stack(characters, axis = 3)

    #Setting up input tensors
    game_tensor = tf.placeholder('float32',[1,172,160,3])
    filters = tf.placeholder('float32',[12,8,3,5])
    
    #Setting up operations
    conv = tf.nn.conv2d(game_tensor, filters, [1,1,1,1], 'SAME')
    
    #Running Session
    with tf.Session() as sess:
        conv_tensor = sess.run(conv,
                 feed_dict={game_tensor: raw_frame, filters: character_filters})
    
    character_locs = {}
    
    #Finding maximum values for convolutions to get character locations
    character_names = ['pac','yellow','red','blue','purple']
    
    #Looping through characters to make sure that locations are accurate
    for i, key in enumerate(character_names):
        if key == 'pac':
            #Using Pacman's previous location to track it more accrately
            location =  np.unravel_index(np.argmax(conv_tensor[0,pacman_track[0]-5:pacman_track[0]+5,pacman_track[1]-5:pacman_track[1]+5,i]),(10,10))
            character_locs[key] = (pacman_track[0]-5 + location[0], pacman_track[1]-5 + location[1])      
        else:
            location = np.unravel_index(np.argmax(conv_tensor[0,:,:,i]),(172,160))
            conv_max = np.max(conv_tensor[0,:,:,i])
            if conv_max <= 200:
                character_locs[key] = (0,0)
            else:
                character_locs[key] = location


    return character_locs

def extract_features(frame, pac_track = (103,79)):    
    """This function takes in a frame and extracts useful features to create a more manageable game state space"""
    character_locs = find_characters(frame, pac_track)
    game_state = np.array([])
    for i in character_locs.keys():
        if i == 'pac':
            game_state = np.append(game_state, character_locs[i])
        else:
            distance = (character_locs['pac'][0] - character_locs[i][0], character_locs['pac'][1] - character_locs[i][1])

            game_state = np.append(game_state, distance)
    return game_state

"""
Testing
env = gym.make('MsPacman-v0')
obs = env.reset()

characters = load_char_pickle()
init = (103,79)

for i in range(200):
    character_locs = find_characters_tf(obs, characters,init)
    print(character_locs)
    #plt.imshow(obs)
    #plt.show()
    action = env.action_space.sample()
    obs = env.step(action)[0]
    init = character_locs['pac']
"""
