import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

env = gym.make('MsPacman-v0')
raw_pixel_obs = env.reset()

print(raw_pixel_obs.shape) #Printing shape of the raw pixel game state
plt.imshow(raw_pixel_obs)

pacman_filter = np.copy(raw_pixel_obs[98:110,75:85])
plt.imshow(pacman_filter) #MsPacman filter to convolve with observation to find MsPacman's position

ghost_f1 = np.copy(raw_pixel_obs[80:92,75:85]) #Ghost 1 Filter
plt.imshow(ghost_f1)

ghost_f2 = np.copy(raw_pixel_obs[50:62,75:85]) #Ghost 2 Filter
plt.imshow(ghost_f2)

#Hacky way of getting the Ghost Filters for the purple and blue ghosts

for i in range(30):
    raw_pixel_obs = env.step(1)[0]
    env.render()

plt.imshow(raw_pixel_obs)
plt.show()

ghost_f3 = np.copy(raw_pixel_obs[80:92,75:85]) #Ghost 3 filter
plt.imshow(ghost_f3)
plt.show()

env.reset()
for i in range(15):
    raw_pixel_obs = env.step(1)[0]
    env.render()

plt.imshow(raw_pixel_obs)
plt.show()

ghost_f4 = np.copy(raw_pixel_obs[80:92,75:85]) #Ghost 4 filter
plt.imshow(ghost_f4)
plt.show()

character_dict = {'pacman': pacman_filter, 'ghost1': ghost_f1,
                    'ghost2': ghost_f2, 'ghost3': ghost_f3,
                    'ghost4': ghost_f4}

pickle.dump(character_dict, open("characters.p","wb"), protocol = 2)
