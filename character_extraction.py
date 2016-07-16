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
import sys
 
def load_char_pickle():
    """Loads character templates"""
    if sys.version[0] == '2':
        return pickle.load(open("characters.p","rb"))
    return pickle.load(open("characters.p","rb"),encoding='latin1') 
 
def normalize_image(image): 
    """Scales images with RGB Channels
    
    args: Image
    returns: Normalized image with respect to each axis seperately
    """ 
    image = image.astype('float32')
    means = np.mean(image,axis = (0,1))
    stds = np.std(image, axis = (0,1))
    for i in range(3):
        m = means[i]
        s = stds[i]
        image[:,:,i] = image[:,:,i] - m 
        image[:,:,i] = image[:,:,i]/s
     
    return image 

def in_range(num, rang):
    return rang[0] < num < rang[1]

class Tracker(object):

    def __init__(self):
        self.tracks = {'pac':(103,79), 'yellow':(86,79),'red':(56,79),'blue':(86,79),'purple':(86,79)}

    def find_characters(self,raw_frame):
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
        pad = 50
        raw_frame = np.pad(raw_frame,((pad,pad),(pad,pad), (0,0)),'constant',constant_values = 0)

        characters = load_char_pickle()
        
        #Formatting Game frame to match TensorFlow Convolution
        raw_frame = np.array([raw_frame])

        character_filters = np.stack(characters, axis = 3)

        #Setting up input tensors
        game_tensor = tf.placeholder('float32',[1,172 + 2*pad,160 + 2*pad,3])
        filters = tf.placeholder('float32',[12,8,3,5])
        
        #Setting up operations
        conv = tf.nn.conv2d(game_tensor, filters, [1,1,1,1], 'SAME')
        
        #Running Session
        with tf.Session() as sess:
            conv_tensor = sess.run(conv,
                     feed_dict={game_tensor: raw_frame, filters: character_filters})
        tf.reset_default_graph()

        character_locs = {}
        
        #Finding maximum values for convolutions to get character locations
        character_names = ['pac','yellow','red','blue','purple']

        correct_values = [(300,500),(1400,1600),(600,800),(500,800),(500,800)]


        #Looping through characters to make sure that locations are accurate
        for i, key in enumerate(character_names):
            window = 15
            #Using Pacman's previous location to track it more accurately
            conv_max = np.max(conv_tensor[0,:,:,i])
            char_track = self.tracks[key]
            possible_jump = False
            if not(in_range(char_track[1], (10,150))) and (in_range(char_track[0],(45,60)) or in_range(char_track[0],(75,110))):
                possible_jump = True

            local_max = np.max(conv_tensor[0,char_track[0]-window+pad:char_track[0]+window+pad,char_track[1]-window+pad:char_track[1]+window+pad,i])
            
            if possible_jump:
                #Portal Case
                location =  np.unravel_index(np.argmax(conv_tensor[0,:,:,i]),(172 + 2*pad,160 + 2*pad))
                location = (location[0] - pad, location[1] - pad)
            
            else:
                if not(in_range(local_max, correct_values[i])) and not(key=='pac'): 
                    #Weird Ghost Case
                    if not(in_range(conv_max, correct_values[i])): 
                        #Disappearing Ghost Case
                        location = self.tracks[key]
                    else:
                        #Lost ghost tracks because of obstacle or dissapearance, increases window size
                        window = window*3
                        tiny_conv = conv_tensor[0, char_track[0]-window+pad:char_track[0]+window+pad, char_track[1]-window+pad:char_track[1]+window + pad, i]
                        location =  np.unravel_index(np.argmax(tiny_conv),(window*2, window*2))
                        location = (char_track[0]-window + location[0], char_track[1]-window + location[1])       
                else:
                    #Normal Ghost Tracking and Pacman Case
                    tiny_conv = conv_tensor[0, char_track[0]-window + pad:char_track[0]+window + pad, char_track[1]-window + pad:char_track[1]+window + pad, i]
                    location =  np.unravel_index(np.argmax(tiny_conv),(window*2, window*2))
                    location = (char_track[0]-window + location[0], char_track[1]-window + location[1])      

            character_locs[key] = location
        
        self.tracks = character_locs

    def extract_features(self, frame):    
        """This function takes in a frame and extracts useful features to create a more manageable game state space"""
        self.find_characters(frame)
        character_locs = self.tracks
        game_state = np.array([])
        for i in character_locs.keys():
            if i == 'pac':
                game_state = np.append(game_state, character_locs[i])
            else:
                distance = (character_locs['pac'][0] - character_locs[i][0], character_locs['pac'][1] - character_locs[i][1])
                
                game_state = np.append(game_state, distance)
        game_state = game_state/172. #Normalizing distances
        return game_state
"""
#Testing
env = gym.make('MsPacman-v0')
obs = env.reset()

tracker = Tracker()

for i in range(200):
    tracker.find_characters(obs)
    character_locs = tracker.tracks
    print(character_locs)
    env.render()
    action = env.action_space.sample()
    obs = env.step(action)[0]"""
