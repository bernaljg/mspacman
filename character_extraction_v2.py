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
import time

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

def calc_luminance(frame):
    return 0.2126*frame[:,:,0] + 0.7152*frame[:,:,1] + 0.0722*frame[:,:,2]

def region_of_interest(frame, location, window_size):
    """Returns the window_size by 20 image around a specific location. Wraps around the edges
    args: array with shape (1,y,x,1)"""
    big_h, big_w = frame[0,:,:,0].shape

    x,y = location[1], location[0]
    half = window_size/2
    h1,h2 = y - half, y + half
    w1,w2 = x - half, x + half

    if half <= y < big_h - half and half <= x < big_w - half: #Mid Frame 
        crop = frame[:,h1:h2,w1:w2,:]

    elif not(half <= y < big_h - half) and not(half <= x < big_w - half): #Corner
        crop = np.zeros((1,window_size,window_size,3))
        if y > 100 and x >100:
            crop[:,0:big_h-h1,0:big_w-w1,:] = frame[:,h1:big_h,w1:big_w,:]
        if y < 100 and x >100:
            crop[:,window_size-h2:window_size,0:big_w-w1,:] = frame[:,0:h2,w1:big_w,:]
        if y > 100 and x <100:
            crop[:,0:big_h-h1,window_size-w2:window_size,:] = frame[:,h1:big_h,0:w2,:]
        if y < 100 and x <100:
            crop[:,window_size-h2:window_size,window_size-w2:window_size,:] = frame[:,0:h2,0:w2,:]
    
    elif not(half <= x < big_w -half): #Edge in X coor, Mid Frame in Y coor
        crop = np.zeros((1,window_size,window_size,3))
        if x < 100:
            crop[:,:,window_size-w2:window_size,:] = frame[:,h1:h2,0:w2,:]
            crop[:,:,0:window_size-w2,:] = frame[:,h1:h2,big_w-window_size+w2:big_w,:]
        if x > 100:
            crop[:,:,0:big_w-w1,:] = frame[:,h1:h2,w1:big_w,:]
            crop[:,:,big_w-w1:window_size,:] = frame[:,h1:h2,0:window_size-w1,:]

    else:                               #Edge in Y coor, Mid Frame in X coor
        crop = np.zeros((1,window_size,window_size,3))
        if y < 100:
            crop[:,window_size-h2:window_size,:,:] = frame[:,0:h2,w1:w2,:]
        if y > 100:
            crop[:,0:big_h-h1,:,:] = frame[:,h1:big_h,w1:w2,:]
         
    return crop

def find_characters(raw_frame, coors, tracking_on): 
    """Finds character locations by using convolution with templates
    args: 
        Raw Frame Data
        Character Template Dictionary
        Pacman Starting Point
        
    returns:
        Dictionary with character locations
    """
    if tracking_on:
        window_size = 50
    else:
        window_size = 150

    #Preprocessing Data 
    raw_frame = raw_frame[:172]
    raw_frame = normalize_image(raw_frame)
    
    #Formatting Game frame to match TensorFlow Convolution
    raw_frame = np.reshape(raw_frame,(1,172,160,3))
    
    #if coors == None:
    frames = []
    for location in coors:
        frames.append(region_of_interest(raw_frame,location,window_size))
    
    characters = load_char_pickle()
    for i in range(5):
        characters[i] = np.reshape(characters[i],(12,8,3,1))


    #Setting up input tensors
    pac_frame = tf.placeholder('float32',[1,window_size,window_size,3])
    y_frame = tf.placeholder('float32',[1,window_size,window_size,3])
    r_frame = tf.placeholder('float32',[1,window_size,window_size,3])
    b_frame = tf.placeholder('float32',[1,window_size,window_size,3])
    p_frame = tf.placeholder('float32',[1,window_size,window_size,3])
    
    pac_fil = tf.placeholder('float32',[12,8,3,1])
    y_fil = tf.placeholder('float32',[12,8,3,1])
    r_fil = tf.placeholder('float32',[12,8,3,1])
    b_fil = tf.placeholder('float32',[12,8,3,1])
    p_fil = tf.placeholder('float32',[12,8,3,1])
    
    #Setting up operations
    pac_conv = tf.nn.conv2d(pac_frame, pac_fil, [1,1,1,1], 'SAME')
    y_conv = tf.nn.conv2d(y_frame, y_fil, [1,1,1,1], 'SAME')
    r_conv = tf.nn.conv2d(r_frame, r_fil, [1,1,1,1], 'SAME')
    b_conv = tf.nn.conv2d(b_frame, b_fil, [1,1,1,1], 'SAME')
    p_conv = tf.nn.conv2d(p_frame, p_fil, [1,1,1,1], 'SAME')

    start = time.time()
    #Running Session
    with tf.Session() as sess:
        pac_conv = sess.run(pac_conv, feed_dict={pac_frame: frames[0], pac_fil: characters[0]})
        y_conv = sess.run(y_conv, feed_dict={y_frame: frames[1], y_fil: characters[1]})
        r_conv = sess.run(r_conv, feed_dict={r_frame: frames[2], r_fil: characters[2]})
        b_conv = sess.run(b_conv, feed_dict={b_frame: frames[3], b_fil: characters[3]})
        p_conv = sess.run(p_conv, feed_dict={p_frame: frames[4], p_fil: characters[4]})
    tf.reset_default_graph()
    conv_time = time.time() - start
    print(conv_time)
    
    #Finding maximum values for convolutions to get character locations
    conv_names = ['pac_conv','y_conv','r_conv','b_conv','p_conv']
    correct_values = [(350,450),(1500,1700),(800,1100),(500,800),(500,800)]

    #Looping through characters to make sure that locations are accurate
    for i, conv in enumerate(conv_names):
        win_loc = np.unravel_index(np.argmax(eval(conv)[0,:,:,0]),(window_size,window_size))
        if 
        real_location = (coors[i][0] + win_loc[0] - window_size/2, coors[i][1] + win_loc[1] - window_size/2)      
        
        #Debug Help
        conv_max = np.max(eval(conv)[0,:,:,0])
        print(conv, conv_max)

        vals = correct_values[i]
        if vals[0] < conv_max < vals[1]:
            coors[i] = location

    return coors

def extract_features(frame, coors):    
    """This function takes in a frame and extracts useful features to create a more manageable game state space"""
    start = time.time()

    coors = [(103,79),(86,79),(56,79),(86,79),(56,79)]
    character_locs = find_characters(frame, coors)
    
    find = time.time() - start
    print("find_time "+ str(find))
    game_state = np.array([])
    for i, char in enumerate(characters):
        if i == 0:
            game_state = np.append(game_state, character_locs[i])
        else:
            distance = (characters[0][0] - char[0], character_locs[0][1] - char[1])

            game_state = np.append(game_state, distance)

    return game_state


#Testing
env = gym.make('MsPacman-v0')
obs = env.reset()

init = [(103,79),(86,79),(56,79)]
coors = None
coors = [(103,79),(86,79),(56,79),(86,79),(86,79)]

for i in range(300):
    coors = find_characters(obs, coors, True)
    print(coors)
    if i > 100:
        plt.imshow(obs)
        plt.show()
    env.render()
    action = env.action_space.sample()
    obs = env.step(action)[0]
