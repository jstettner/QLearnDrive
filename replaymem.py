#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random

class ReplayMemory(object):
    def __init__(self, sample_size):
        super(ReplayMemory, self).__init__()
        self.rm = []
        self.sample_size = sample_size

    def ins(s, a, r, s_prime):
        self.rm.append([s, a, r, s_prime])

    def getMem():
        return self.rm

    def sample():
        if(len(rm) == 0 or sample_size > len(self.rm)):
            return -1
        else:
            ran = random.randint(0, len(self.rm)-sample_size)
            return self.rm[ran : ran + sample_size]
