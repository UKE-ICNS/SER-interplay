# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 13:24:39 2021

@author: Maria

torch version of SER with a buffer to save memory and varying Cs
"""
import numpy as np
import torch as th

def SERmodel_multneuro_buf_c(Cs,T,ia=1): 
    """
    SER network simulation for multiple neurons in one region

    Cs            = Matrix of coupling (NxN) between pairs of neurons (can be directed)
    T            = Total time of simulated activity
    ia           = Initial condition setting. It can be a number representing the number
                 of excited nodes (the remaining nodes are splitted in two equal
                 size cohorts of susceptible and refractory nodes) or can be
                 a vector describing the initial state of each region

    Convention is:
          - susceptible node =  0
          - excited node     =  1
          - refractory node  = -1

    as a result, gives an activity for a particular matrices Cs under the SER dynamics
    for a given initial condition setting.
    """

    N = np.shape(Cs)[1] #number of regions
    ia_ten = th.tensor(ia, dtype=th.int8).cuda() #make a tensor
    Cs_ten = th.tensor(Cs, dtype=th.int8).cuda() #make a tensor
    states = ia_ten.shape[0] 
    graphs = Cs_ten.shape[0] 
    buf = 10 #how large is the frame we want to save in memory

    y = th.zeros((graphs,states,N,buf), dtype=th.int8).cuda() #initialize phase timeseries for one cycle

    ##Initialization
    ia_ten_per = ia_ten.unsqueeze(0).expand(graphs,-1,-1) #expand ia_ten for all Cs
    y[...,0] = ia_ten_per

    Cs_per = Cs_ten.unsqueeze(2).expand(-1,-1,y.shape[1],-1) #expand C for all initial conditions

    ##SER updates
    #generating the buffer
    for t in range(buf-1):

        y[...,t+1][y[...,t]==1]=-1

        y[...,t+1][th.mul(y[...,t]==0, (th.sum(Cs_per*(( y[...,t]==1)[:,None,:]),3)>0).permute(0,2,1))]=1
    

    for t in range(T-buf): 
        # updates for ser model
        y = th.roll(y, shifts=(0,0,0,-1), dims=(0,1,2,3))

        y[...,-1]=0

        y[...,-1][y[...,-2]==1]=-1

        y[...,-1][th.mul(y[...,-2]==0, (th.sum(Cs_per*(( y[...,-2]==1)[:,None,:]),3)>0).permute(0,2,1))]=1

    fin=y
    
    return fin 
