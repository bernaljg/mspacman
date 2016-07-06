import tensorflow as tf 
import gym 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp 
from sklearn import preprocessing 
import scipy.ndimage 
import pickle 
 
def load_char_pickle(): 
    return pickle.load(open("characters.p","rb")) 
 
def scale_image(image): 
    """Scales images with RGB Channels""" 
    image = image.astype('float32') 
 
    for i in range(3): 
        image[:,:,i] = preprocessing.scale(image[:,:,i])  
     
    return image 
 
def find_characters_tf(raw_frame, characters): 
 
    #Preprocessing Data 
    raw_frame = raw_frame[:172]
    raw_frame = scale_image(raw_frame) 
    
    for i,image in enumerate(characters): 
        characters[i] = scale_image(image)
    
    #Formatting Game frame to match TensorFlow Convolution
    raw_frame = np.array([raw_frame])

    character_filters = np.stack(characters, axis = 3)

    #Setting up input tensors
    game_tensor = tf.placeholder('float32',[1,172,160,3])
    filters = tf.placeholder('float32',[12,8,3,10])
    
    #Setting up operations
    conv = tf.nn.conv2d(game_tensor, filters, [1,1,1,1], 'SAME')
    
    #Running Session
    with tf.Session() as sess:
        conv_tensor = sess.run(conv,
                 feed_dict={game_tensor: raw_frame, filters: character_filters})
    
    character_locs = []
    
    #Finding maximum values for convolutions to get character locations
    for i in range(5): #Finds best filter to use
        conv_maxs = [np.max(conv_tensor[0,:,:,i*2]),np.max(conv_tensor[0,:,:,i*2+1])]
        char_orientation = np.argmax(conv_maxs)
        conv_max = np.max(conv_maxs)
        print(conv_max)
        location = np.unravel_index(np.argmax(conv_tensor[0,:,:,i*2+char_orientation]),(172,160))
        
        if location in character_locs:
            pass
        else:
            character_locs.append(location)

    return character_locs

env = gym.make('MsPacman-v0')
obs = env.reset()

characters = load_char_pickle()

frames = []
character_list = []
env.reset()
for i in range(10):
    character_locs = find_characters_tf(obs, characters)
    character_list.append(character_locs)
    print(character_locs)
    plt.imshow(obs)
    plt.show()
    
    action = env.action_space.sample()
    obs = env.step(action)[0]
    frames.append(obs)