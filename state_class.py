# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 20:17:19 2018

@author: Alan
"""

import numpy as np
import pandas as pd

indexes = pd.read_csv('input_data/index_time_series.csv', sep = ';', decimal = ',')

class State:
    
    def __init__(self, asset, liability, historical_return):
        self.asset = asset
        self.liability = liability
        self.historical_return = historical_return
        self.initial_asset = asset
        self.initial_liability = liability
        self.initial_historical_return = historical_return
        
    def step(self, action):
        # Decidir se a random sample da normal multivariada será com os dados 
        # históricos iniciais, ou se vai incluir simulações
        
        sim_return = np.random.multivariate_normal(np.mean(self.historical_return, axis = 0), 
                                                   np.cov(self.historical_return, rowvar = False))
        
        # sim_return = np.random.multivariate_normal(np.mean(self.initial_historical_return, axis = 0), 
        #                                            np.cov(self.initial_historical_return, rowvar = False))
                
        alocation = self.asset * action * (1 + sim_return / 100)
        alocation[0] -= self.liability[0]
        if alocation[0] < 0:
            reward = -np.sum(self.liability)
        elif (alocation[0] >= 0) & (np.sum(self.liability) == 0):
            reward = 1
        else:
            reward = 0
            
        self.asset = np.sum(alocation)
        self.liability = self.liability[1:] * (1 + sim_return[1] / 100)
        self.liability.append(0)
        self.historical_return = np.vstack((self.historical_return, sim_return))
        
        return reward
    
    def reset(self):
        self.asset = self.initial_asset
        self.liability = self.initial_liability
        self.historical_return = self.initial_historical_return
