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
import pickle
     
class Taxi():
    
    def __init__(self):
        
        #make the enviroment
        self.env = gym.make('Taxi-v3', render_mode='rgb_array')
        self.state, info = self.env.reset()
        self.steps = 0

        #open policy matrix
        with open('code/model/taxi_qmatrix.pkl', 'rb') as file:
            self.qmatrix = pickle.load(file)

    def select_action(self, action):
        
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        self.state = observation

        #Print the output after a step
        print("\n\n\nStep number:", self.steps)
        print("\nAction:          ---> ", action)
        print("\nObservation:     ---> ", observation)
        print("\nReward:          ---> ", reward)
        print("\nTerminated:      ---> ", terminated)
        print("\nTruncated:       ---> ", truncated)
        print("\nInfo:            ---> ", info)
        
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