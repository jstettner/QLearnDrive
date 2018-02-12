#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import gym
import universe
import numpy as np

from replaymem import ReplayMemory
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Conv3D
from keras.layers import Dense, Dropout, Flatten

def main():
    env = gym.make('flashgames.NeonRace2-v0')
    env.configure(remotes=1)

    observation_n = env.reset()

    '''
    define keybinds and list of three possible actions
    '''
    left = [('KeyEvent', 'ArrowLeft', True),('KeyEvent', 'ArrowRight', False),('KeyEvent', 'ArrowUp', True),('KeyEvent', 'ArrowDown', False)]
    right = [('KeyEvent', 'ArrowLeft', False),('KeyEvent', 'ArrowRight', True),('KeyEvent', 'ArrowUp', True),('KeyEvent', 'ArrowDown', False)]
    forward = [('KeyEvent', 'ArrowLeft', False),('KeyEvent', 'ArrowRight', False),('KeyEvent', 'ArrowUp', True),('KeyEvent', 'ArrowDown', False)]

    actions = [left, right, forward]

    '''
    let Q be the cnn to determine the quality of each action
    '''
    Q = Sequential()
    Q.add(Conv3D(32, kernel_size=(5,5,3), strides=5,
                 activation='relu',
                 input_shape=(768, 1024, 3, 1)))
    Q.add(Conv3D(64, kernel_size=(5,5,1), strides=3,
                 activation='relu',
                 input_shape=(255, 340, 1, 32)))
    Q.add(Conv3D(64, kernel_size=(4,4,1), strides=2,
                 activation='relu',
                 input_shape=(50, 67, 1, 64)))
    Q.add(Flatten())
    Q.add(Dense(1000, activation='relu', input_shape=(49152,)))
    # Q.add(Dropout(0.2, input_shape=(1000,)))
    Q.add(Dense(512, activation='relu', input_shape=(1000,)))
    # Q.add(Dropout(0.2, input_shape=(512,)))
    Q.add(Dense(128, activation='sigmoid', input_shape=(512,)))
    Q.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
    #
    # '''
    # epsilon determines the probability that the agent acts randomly to avoid local mins
    # '''
    # eps = 1
    #
    # rm = ReplayMemory(4)
    # done_n = [False]
    #
    # while(done_n[0] == False):
    #     ran = random.randint(0, eps)
    #
    #     if(observation_n[0] != None):
    #         observation = np.expand_dims(observation_n[0]['vision'], axis=0)
    #         '''
    #         observation shape of (1, 768, 1024, 3)
    #         first dim is just to format for 2Dcnv layer 1
    #         '''
    #
    #         if(ran == 0):
    #             action_n = [random.choice(actions) for ob in observation_n]
    #         else:
    #             action_n = [actions[np.argmax(Q.predict(observation))] for ob in observation_n]
    #             print(action_n)
    #     else: # random actions during game load
    #         action_n = [random.choice(actions) for ob in observation_n]
    #
    #     s_prime, reward_n, done_n, info = env.step(action_n) # execute a and retrieve s' and r
    #
    #     rm.ins(observation_n, action_n, reward_n, s_prime) # add to replay memory
    #
    #     observation_n = s_prime
    #     env.render()

if __name__ == '__main__':
    main()
