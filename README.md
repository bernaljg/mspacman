This repo contains an artificial intelligence agent that uses Deep Reinforcement Learning to learn how to play MsPacman having access to only the game pixel values.

extract_template.ipyb is a notebook that I used to create the characters.p pickle file which contains the character filters that are used to find the characters from raw pixel values.

character_extract.py extracts characters from raw pixel values by convolving filters and tracking the characters throughout the game

pacman_agent.py contains two classes
  Pacman Agent class is the class that represents the Pacman player. It has training and playing methods.
  DRL Model is a class that stores the weights used in the Deep Q learning algorithm. Each Pacman Agent instance has a DRL Model instance.
  
The trained model is stored in a pickle file called trained_model.p

Running train_pacman.py trains a model for the amount of epochs specified in the file.

Running play_pacman.py plays and renders the game using the current trained_model.p
