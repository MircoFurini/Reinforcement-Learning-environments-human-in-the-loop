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
     
class FrozenLake():
    
    def __init__(self, slippery, desc):
        
        #make the enviroment
        self.env = gym.make('FrozenLake-v1', desc=desc, map_name='4x4', is_slippery=slippery, render_mode='rgb_array')
        self.state, info = self.env.reset()
        self.steps = 0
        self.state = 0
      
    def select_action(self, action):
        
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.state = observation
        #Print the output after a step
        #in info prob is 1 if is_slippery=False else prob is 1/3, also in all perpendicol directions
        print("\n\n\nStep number:", self.steps)
        print("\nAction: 0 left, 1 down, 2 right, 3 up                     ---> ", action)
        print("\nObservation: position = current_row * nrows + current_col ---> ", observation)
        print("\nReward: 1 only if goal is reached                         ---> ", reward)
        print("\nTerminated: player into a hole o reaches the goal         ---> ", terminated)
        print("\nTruncated:                                                ---> ", truncated)
        print("\nInfo: transition probability                              ---> ", info)
        
        self.steps+=1 #increment step numer
        
        if terminated or truncated:
            #self.env.reset()
            #self.steps = 0
            pass

        return reward, self.steps, terminated, truncated, observation 

    def restart_env(self):
        
        self.state, info = self.env.reset()
        self.steps = 0
    
    def render_image(self):
        
        image_rgb = self.env.render()
        image = Image.fromarray(image_rgb)
        
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