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

BATCH_SIZE = 64
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
Q.add(Dense(512, activation='relu', input_shape=(1000,)))
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
        return np.argmax(Q.predict(state))
    else:
        return random.randint(0,2)

def optimize_model():
    print('Starting optimization')

    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    x_train = []
    y_train = []

    for i in range(len(batch[0])):
        ss = batch[0][i]
        aa = batch[1][i]
        if(batch[2][i] != None):
            ss_p = np.expand_dims(np.expand_dims(batch[2][i][0]['vision'], axis=3), axis=0)
        rr = batch[3][i]

        tt = rr
        if(batch[2][i] != None):
            tt = rr + GAMMA*np.amax(Q.predict(ss_p))

        x_train.append(ss)
        y_data = np.zeros(3)
        y_data[aa] = tt
        y_train.append(y_data)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print(x_train)
    print(y_train)
    Q.fit(x=x_train, y=y_train, batch_size=2, epochs=2, verbose=1)

episodes = 5
def main():
    env = gym.make(GAME)
    env.configure(remotes='vnc://localhost:5900+15900')
    # env.configure(remotes=1)
    observation_n = env.reset()

    e = 0
    while(e <= episodes):
        if(observation_n[0] != None):
            e+=1
            print('Entered')
            for time_t in range(500):
                observation = np.expand_dims(observation_n[0]['vision'], axis=3)
                action = select_action(np.expand_dims(observation, axis=0))

                next_state, reward_n, done_n, info = env.step([ACTIONS[action]])
                env.render()

                if(done_n[0]):
                    memory.ins(observation, action, [None], reward_n)
                    print("episode: {}/{}, score: {}"
                        .format(e, episodes, time_t))
                    observation_n = env.reset()
                    break
                else:
                    memory.ins(observation, action, next_state, reward_n)
            else:
                print("Game Restart")
                observation_n = env.reset()
                optimize_model()
        else:
            action_n = [random.choice(ACTIONS)]
        next_state, reward_n, done_n, info = env.step(action_n)

        observation_n = next_state
        env.render()

if __name__ == '__main__':
    main()
