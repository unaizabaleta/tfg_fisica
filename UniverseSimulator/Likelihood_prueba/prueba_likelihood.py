#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 22:01:07 2022

@author: jesus
"""
import numpy as np

class simulator():
    
    def __init__(self, X, params):
        self.X = X
        self.params = params
    
    def evaluation(self):
        Y = np.array(self.X) @ np.array(self.params)
        return Y