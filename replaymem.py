#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        super(ReplayMemory, self).__init__()
        self.capacity = capacity
        self.rm = []
        self.position = 0

    def ins(self, *args):
        if len(self.rm) < self.capacity:
            self.rm.append(None)
        self.rm[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def getMem(self):
        return self.rm

    def sample(self, batch_size):
        if(len(self.rm) == 0 or batch_size > len(self.rm)):
            return None
        else:
            return random.sample(self.rm, batch_size)

    def __len__(self):
        return len(self.rm)
