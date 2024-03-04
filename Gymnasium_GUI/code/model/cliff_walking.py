import gymnasium as gym 
from collections import defaultdict
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import *
from PIL import Image,ImageTk
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
from pathlib import Path
import pygame
import numpy as np
from io import BytesIO
     
class CliffWalking():
    
    def __init__(self):
        
        #make the enviroment
        self.env = gym.make('CliffWalking-v0', render_mode='rgb_array')
        self.env.reset()
        self.steps = 0
        self.state = 36
        self.qmatrix = [[-7.85551867e+00, -7.61724297e+00, -7.61724297e+00, -7.85551867e+00], 
                [-7.61724297e+00, -7.35249219e+00, -7.35249219e+00, -7.85551867e+00],
                [-7.35249219e+00, -7.05832465e+00, -7.05832465e+00, -7.61724297e+00],
                [-7.05832465e+00, -6.73147184e+00, -6.73147184e+00, -7.35249219e+00],
                [-6.73147184e+00, -6.36830204e+00, -6.36830204e+00, -7.05832465e+00],
                [-6.36830204e+00, -5.96478004e+00, -5.96478004e+00, -6.73147184e+00],
                [-5.96478004e+00, -5.51642227e+00, -5.51642227e+00, -6.36830204e+00],
                [-5.51642227e+00, -5.01824697e+00, -5.01824697e+00, -5.96478004e+00],
                [-5.01824697e+00, -4.46471885e+00, -4.46471885e+00, -5.51642227e+00],
                [-4.46471885e+00, -3.84968761e+00, -3.84968761e+00, -5.01824697e+00],
                [-3.84968761e+00, -3.16631957e+00, -3.16631957e+00, -4.46471885e+00],
                [-3.16631957e+00, -3.16631957e+00, -2.40702175e+00, -3.84968761e+00],
                [-7.85551867e+00, -7.35249219e+00, -7.35249219e+00, -7.61724297e+00],
                [-7.61724297e+00, -7.05832465e+00, -7.05832465e+00, -7.61724297e+00],
                [-7.35249219e+00, -6.73147184e+00, -6.73147184e+00, -7.35249219e+00],
                [-7.05832465e+00, -6.36830204e+00, -6.36830204e+00, -7.05832465e+00],
                [-6.73147184e+00, -5.96478004e+00, -5.96478004e+00, -6.73147184e+00],
                [-6.36830204e+00, -5.51642227e+00, -5.51642227e+00, -6.36830204e+00],
                [-5.96478004e+00, -5.01824697e+00, -5.01824697e+00, -5.96478004e+00],
                [-5.51642227e+00, -4.46471885e+00, -4.46471885e+00, -5.51642227e+00],
                [-5.01824697e+00, -3.84968761e+00, -3.84968761e+00, -5.01824697e+00],
                [-4.46471885e+00, -3.16631957e+00, -3.16631957e+00, -4.46471885e+00],
                [-3.84968761e+00, -2.40702175e+00, -2.40702175e+00, -3.84968761e+00],
                [-3.16631957e+00, -2.40702175e+00, -1.56335750e+00, -3.16631957e+00],
                [-7.61724297e+00, -7.05832465e+00, -7.61724297e+00, -7.35249219e+00],
                [-7.35249219e+00, -6.73147184e+00, -1.06617243e+02, -7.35249219e+00],
                [-7.05832465e+00, -6.36830204e+00, -1.06617243e+02, -7.05832465e+00],
                [-6.73147184e+00, -5.96478004e+00, -1.06617243e+02, -6.73147184e+00],
                [-6.36830204e+00, -5.51642227e+00, -1.06617243e+02, -6.36830204e+00],
                [-5.96478004e+00, -5.01824697e+00, -1.06617243e+02, -5.96478004e+00],
                [-5.51642227e+00, -4.46471885e+00, -1.06617243e+02, -5.51642227e+00],
                [-5.01824697e+00, -3.84968761e+00, -1.06617243e+02, -5.01824697e+00],
                [-4.46471885e+00, -3.16631957e+00, -1.06617243e+02, -4.46471885e+00],
                [-3.84968761e+00, -2.40702175e+00, -1.06617243e+02, -3.84968761e+00],
                [-3.16631957e+00, -1.56335750e+00, -1.06617243e+02, -3.16631957e+00],
                [-2.40702175e+00, -1.56335750e+00, -6.25952773e-01, -2.40702175e+00],
                [-7.35249219e+00, -1.06617243e+02, -7.61724297e+00, -7.61724297e+00],
                [5.83894835e-01, 3.94479149e-01, 6.66018745e-01, 1.92758645e-01],
                [6.87078604e-01, 9.34703192e-01, 2.46112810e-01, 2.28638205e-01],
                [4.68171658e-01, 6.80211405e-01, 5.70693059e-01, 2.41732188e-02],
                [1.18333533e-01, 3.07042783e-03, 6.85496996e-01, 5.73882802e-01],
                [1.24286676e-01, 9.29756805e-01, 9.75394115e-01, 5.72413537e-02],
                [8.05397839e-01, 6.30316122e-01, 2.82072923e-01, 1.41020819e-01],
                [4.22786132e-01, 9.85257585e-01, 5.83754773e-02, 3.62711803e-02],
                [4.48221065e-02, 4.30749860e-01, 4.08692284e-01, 9.42912802e-01],
                [7.13847215e-01, 5.81284469e-03, 3.38794742e-01, 3.98052312e-01],
                [1.18637689e-01, 5.41007228e-01, 3.16026737e-01, 8.19301704e-01],
                [3.75225521e-01, 1.74308315e-01, 1.73685389e-01, 4.15608030e-01]]


    def select_action(self, action):
        
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        self.state = observation

        #Print the output after a step
        print("\n\n\nStep number:", self.steps)
        print("\nAction:                                           ---> ", action)
        print("\nObservation: current_row * nrows + current_col    ---> ", observation)
        print("\nReward: -1 per step, -100 into the cliff          ---> ", reward)
        print("\nTerminated: state 47(reach goal)                  ---> ", terminated)
        print("\nTruncated:  step_limit                            ---> ", truncated)
        print("\nInfo: p-probability(always 1.00)                  ---> ", info)
        
        self.steps+=1 #increment step numer

        if terminated or truncated:
            #self.env.reset()
            #self.steps = 0
            pass

        return reward, self.steps, terminated, truncated, observation 

    def restart_env(self):
        
        self.env.reset()
        self.steps = 0
        self.state = 36
    
    
    def render_image(self):
        
        image_rgb = self.env.render()
        image = Image.fromarray(image_rgb)

        #image_rgb = cliffWalking.env.render()
        #image = Image.fromarray(image_rgb)

        
        #image pygame
        #image = Image.fromarray(image,'RGB')
        #PIL_image = Image.fromarray(np.uint8(image)).convert('RGB')
        
        return image   
    
    def get_steps(self):
        
        return self.steps
    
    def get_env(self):
         
        return self.env 
    
    def close_env(self):
        
        self.env.close()

    def get_state(self):

        return self.state

    def get_qmatrix(self):
        
        return self.qmatrix
    
    def learnedAction(self, state):
            
        return np.random.choice(np.where(self.qmatrix[state]==np.max(self.qmatrix[state]))[0])