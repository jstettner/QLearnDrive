#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import gym
import universe
import numpy as np
import math

from replaymem import ReplayMemory, Transition
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Conv3D
from keras.layers import Dense, Dropout, Flatten

GAME = 'flashgames.NeonRace2-v0'

LEFT = [('KeyEvent', 'ArrowLeft', True),('KeyEvent', 'ArrowRight', False),('KeyEvent', 'ArrowUp', True),('KeyEvent', 'ArrowDown', False)]
RIGHT = [('KeyEvent', 'ArrowLeft', False),('KeyEvent', 'ArrowRight', True),('KeyEvent', 'ArrowUp', True),('KeyEvent', 'ArrowDown', False)]
FORWARD = [('KeyEvent', 'ArrowLeft', False),('KeyEvent', 'ArrowRight', False),('KeyEvent', 'ArrowUp', True),('KeyEvent', 'ArrowDown', False)]

ACTIONS = [LEFT, RIGHT, FORWARD]

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

memory = ReplayMemory(10000)

done_n = [False]

steps_done = 0

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
Q.add(Dense(128, activation='relu', input_shape=(512,)))
Q.add(Dense(3, activation='linear', input_shape=(128,)))
Q.compile(loss='categorical_crossentropy',
        optimizer=RMSprop(),
        metrics=['accuracy'])

def select_action(state):
    global steps_done
    global Q
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        return [ACTIONS[np.argmax(Q.predict(state))]]
    else:
        return [random.choice(ACTIONS)]

last_sync = 0

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

def main():
    env = gym.make(GAME)
    env.configure(remotes=1)

    observation_n = env.reset()

    while(True):
        if(observation_n[0] != None):
            observation = np.expand_dims(np.expand_dims(observation_n[0]['vision'], axis=3), axis=0)
            action_n = select_action(observation)
            memory.ins(observation, action_n, next_state, reward_n)
            optimize_model()
        else:
            action_n = [random.choice(ACTIONS)]

        next_state, reward_n, done_n, info = env.step(action_n)

        observation_n = next_state
        env.render()

if __name__ == '__main__':
    main()
