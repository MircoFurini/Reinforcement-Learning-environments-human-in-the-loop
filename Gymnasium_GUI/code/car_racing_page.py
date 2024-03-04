import tkinter as tk
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
import gymnasium as gym 
from collections import defaultdict
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from PIL import Image,ImageTk
from pathlib import Path
import pygame
import numpy as np
import importlib
from datetime import datetime
from gymnasium.utils.play import play


class CarRacingPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#FFFFFF")
        self.controller = controller

        self.tot_reward = 0
        self.check_field = False #check field setting, enable arrows 
        self.episode_number = 0
        self.episodes_arr = [] #store the number of episode
        self.episodes_reward = [] #save the tot_reward of every episode
        self.episode_steps = [] #save the number of steps of each episode
        self.terminated = False
        self.truncated = False
        self.steps = 0


        #####title
        #canvas window
        canvas = Canvas(
            self,
            bg = "#FFFFFF",
            height = 100,
            width = 1920,
            bd = 0,
            highlightthickness = 0,
            relief = "ridge"
        )

        canvas.place(x = 0, y = 0)

        canvas.create_rectangle(
            0.0,
            0.0,
            1920.0,
            100.0,
            fill="#8398E3",
            outline="")

        canvas.create_text(
            708.0,
            19.0,
            anchor="nw",
            text="CAR RACING DASHBOARD",
            fill="#FFFFFF",
            font=("Inter Bold", 40 * -1)
        )


        #######################buttons
        ##comunication_buttons
        
        self.label_comunication = Label(
                                self,
                                text="",
                                font=("Inter", 15, "bold"),
                                fg="#8398E3",
                                bg="#FFFFFF")
        self.label_comunication.place(x=40, y=320)

        ##preliminary buttons
        #episode & save button
        
        self.button_new_episode = Button(
            self,
            text="HUMAN",
            font = ("Inter", 10, "bold"),
            bg="#F38630",
            fg="#FFFFFF",
            command=self.button_new_episode_clicked,
        )
        #display new buttons
        self.button_new_episode.place(
            x=40.0,
            y=110.0,
            width=240.0,
            height=70.0
            )
                

        self.button_save = Button(
            self,
            text="SAVE EPISODE REPORT",
            font = ("Inter", 10, "bold"),
            bg="#F38630",
            fg="#FFFFFF",
            command=self.button_save_clicked,
        )
        self.button_save.place(
            x=330.0,
            y=110.0,
            width=240.0,
            height=70.0
        )

        ##buttons
        self.button_home = Button(
            self,
            text="HOME",
            font = ("Inter", 10, "bold"),
            bg="#F38630",
            fg="#FFFFFF",
            command=self.button_home_clicked,
        )
        self.button_home.place(
            x=43.0,
            y=880.0,
            width=240.0,
            height=70.0
        )


        ######graph matplotlib
        # Initialization graph
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8))
        plt.subplots_adjust(hspace=0.5, wspace=0.2)

        self.ax1 = self.axs[0, 0]
        self.ax2 = self.axs[0, 1]
        self.ax3 = self.axs[1, 0]
        self.ax4 = self.axs[1, 1]

        self.x_data_tot_reward = []  
        self.y_data_tot_reward = []  
        self.line_tot_reward, = self.ax1.plot(self.x_data_tot_reward, self.y_data_tot_reward, 'r-')

        self.x_data_rew = [] 
        self.y_data_rew = []  
        self.line_rew, = self.ax2.plot(self.x_data_rew, self.y_data_rew, 'bo')

        self.x_data_steer = []  
        self.y_data_steer = []  
        self.line_steer, = self.ax3.plot(self.x_data_steer, self.y_data_steer, 'go')
        self.ax3.set_ylim(-2,2)

        self.x_data_gas = []  
        self.y_data_gas = []  
        self.line_gas, = self.ax4.plot(self.x_data_gas, self.y_data_gas, 'go')

        self.x_data_Break = []  
        self.y_data_Break = []  
        self.line_Break, = self.ax4.plot(self.x_data_Break, self.y_data_Break, 'ro')

        # Configuration graph
        self.ax1.set_title('Total reward')
        self.ax1.set_xlabel('step')
        self.ax1.set_ylabel('total reward')

        self.ax2.set_title('Reward')
        self.ax2.set_xlabel('step')
        self.ax2.set_ylabel('reward')

        self.ax3.set_yticks([-1,0,1])
        self.ax3.set_yticklabels(['left', 'noop', 'right'])
        self.ax3.set_title('Steer')
        self.ax3.set_xlabel('step')
        self.ax3.set_ylabel('steer')

        self.ax4.set_title('Gas and Breaking')
        self.ax4.set_xlabel('step')
        self.ax4.set_ylabel('gas/breaking')

        self.ax2.set_xlim(-0.5, 10)
        self.ax3.set_xlim(-0.5, 10)
        self.ax4.set_xlim(-0.5, 10)

        ##graph
        self.left_frame = tk.Frame(self, width=800, height=650)
        self.left_frame.place(x=800, y=200)
            
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0)


    ##################### FUNCTIONS ####################
    
    def button_new_episode_clicked(self):
        
        #reset variables
        self.steps = 0
        self.tot_reward = 0
        
        ##Reset graph for new episode
        self.x_data_tot_reward = []  
        self.y_data_tot_reward = []  
        self.line_tot_reward.set_xdata(self.x_data_tot_reward)
        self.line_tot_reward.set_ydata(self.y_data_tot_reward)

        self.x_data_rew = []  
        self.y_data_rew = [] 
        self.line_rew.set_xdata(self.x_data_rew)
        self.line_rew.set_ydata(self.y_data_rew) 

        self.x_data_steer = []  
        self.y_data_steer = [] 
        self.line_steer.set_xdata(self.x_data_steer)
        self.line_steer.set_ydata(self.y_data_steer)  

        self.x_data_gas = []  
        self.y_data_gas = []  
        self.line_gas.set_xdata(self.x_data_gas)
        self.line_gas.set_ydata(self.y_data_gas)

        self.x_data_Break = []  
        self.y_data_Break = [] 
        self.line_Break.set_xdata(self.x_data_Break)
        self.line_Break.set_ydata(self.y_data_Break) 
        
        #delete the preliminary widgets
        self.button_save.place_forget()
        self.button_new_episode.place_forget()
        self.label_comunication.place(x=70, y=320)
        self.label_comunication.config(text = "Use \'wasd\' keys \nto control the car. \n\nClose the game window \nto start a new episode", fg ="#8398E3")
        
        #start game
        play(gym.make('CarRacing-v2', autoreset = False, render_mode="rgb_array"),callback=self.callback, keys_to_action={
            "w": np.array([0, 0.7, 0]),
            "a": np.array([-1, 0, 0]),
            "s": np.array([0, 0, 0.5]),
            "d": np.array([1, 0, 0]),
            "wa": np.array([-1, 0.7, 0]),
            "dw": np.array([1, 0.7, 0]),
            "ds": np.array([1, 0, 0.5]),
            "as": np.array([-1, 0, 0.5]),
        }, noop=np.array([0, 0, 0]))
        
        #place again the buttons
        
        self.label_comunication.place_forget()
        
        self.button_new_episode.place(
            x=40.0,
            y=110.0,
            width=240.0,
            height=70.0
        )
        
        self.button_save.place(
            x=330.0,
            y=110.0,
            width=240.0,
            height=70.0
        )

    def callback(self, obs_t, obs_tp1, action, rew, terminated, truncated, info):

        self.tot_reward += rew
        
        #update graph
        self.x_data_tot_reward.append(self.steps)
        self.y_data_tot_reward.append(self.tot_reward)
        self.line_tot_reward.set_ydata(self.y_data_tot_reward)
        self.line_tot_reward.set_xdata(self.x_data_tot_reward)
        self.ax1.relim()  # Ridisegna i limiti dell'asse
        self.ax1.autoscale_view(True, True, True)  # Ridisegna l'asse in base ai nuovi dati
        self.ax1.set_xlabel(f'Step: {self.steps}')

        self.x_data_rew.append(self.steps)
        self.x_data_rew = self.x_data_rew[-10:]  # mantieni solo gli ultimi 100 punti
        self.y_data_rew.append(rew)
        self.y_data_rew = self.y_data_rew[-10:]  # mantieni solo gli ultimi 100 punti
        self.line_rew.set_ydata(self.y_data_rew)
        self.line_rew.set_xdata(self.x_data_rew)
        self.ax2.relim()  # Ridisegna i limiti dell'asse
        self.ax2.autoscale_view(True, True, True)  # Ridisegna l'asse in base ai nuovi dati
        self.ax2.set_xlabel(f'Step: {self.steps}')

        self.x_data_steer.append(self.steps)
        self.x_data_steer= self.x_data_steer[-10:]  
        self.y_data_steer.append(action[0])
        self.y_data_steer= self.y_data_steer[-10:]  
        self.line_steer.set_ydata(self.y_data_steer)
        self.line_steer.set_xdata(self.x_data_steer)
        self.ax3.relim()  # Ridisegna i limiti dell'asse
        self.ax3.autoscale_view(True, True, True)  # Ridisegna l'asse in base ai nuovi dati
        self.ax3.set_xlabel(f'Step: {self.steps}')

        self.x_data_gas.append(self.steps)
        self.x_data_gas= self.x_data_gas[-10:]  
        self.y_data_gas.append(action[1])
        self.y_data_gas= self.y_data_gas[-10:]  
        self.line_gas.set_ydata(self.y_data_gas)
        self.line_gas.set_xdata(self.x_data_gas)

        self.x_data_Break.append(self.steps)
        self.x_data_Break= self.x_data_Break[-10:]  
        self.y_data_Break.append(action[2])
        self.y_data_Break= self.y_data_Break[-10:]  
        self.line_Break.set_ydata(self.y_data_Break)
        self.line_Break.set_xdata(self.x_data_Break)
        self.ax4.relim()  # Ridisegna i limiti dell'asse
        self.ax4.autoscale_view(True, True, True)  # Ridisegna l'asse in base ai nuovi dati
        self.ax4.set_xlabel(f'Step: {self.steps}')
        self.ax4.legend([self.line_gas, self.line_Break],['gas','break'])

        if self.steps <= 9:
            self.ax2.set_xlim(-0.5, 10)
            self.ax3.set_xlim(-0.5, 10)
            self.ax4.set_xlim(-0.5, 10)
        
        else:
            self.ax2.set_xlim(self.steps -10, self.steps+1)
            self.ax3.set_xlim(self.steps -10, self.steps+1)
            self.ax4.set_xlim(self.steps -10, self.steps+1)
        
        plt.draw()
        self.canvas.draw()
        self.update()
        
        #Manage the episode 
        if terminated or truncated:
            
            self.episodes_reward.append(self.tot_reward)
            self.episodes_arr.append(self.episode_number)
            self.episode_steps.append(self.steps)
            print(f"Episode tot_reward: {self.tot_reward}")
            self.episode_number+=1
            self.tot_reward = 0
            self.steps = 0

            ##Reset graph for new episode
            self.x_data_tot_reward = []  
            self.y_data_tot_reward = []  
            self.line_tot_reward.set_xdata(self.x_data_tot_reward)
            self.line_tot_reward.set_ydata(self.y_data_tot_reward)

            self.x_data_rew = []  
            self.y_data_rew = [] 
            self.line_rew.set_xdata(self.x_data_rew)
            self.line_rew.set_ydata(self.y_data_rew) 

            self.x_data_steer = []  
            self.y_data_steer = [] 
            self.line_steer.set_xdata(self.x_data_steer)
            self.line_steer.set_ydata(self.y_data_steer)  

            self.x_data_gas = []  
            self.y_data_gas = []  
            self.line_gas.set_xdata(self.x_data_gas)
            self.line_gas.set_ydata(self.y_data_gas)

            self.x_data_Break = []  
            self.y_data_Break = [] 
            self.line_Break.set_xdata(self.x_data_Break)
            self.line_Break.set_ydata(self.y_data_Break) 


        self.steps+=1
    
    #button save & new episode
    def button_save_clicked(self): #save data and restart episode
            
        if self.episodes_arr == []:
            self.label_comunication.place(x=70, y=320)
            self.label_comunication.config(text=f"No data to save! Restart a match!", fg = "red")
        else:
            df = pd.DataFrame({'episode': self.episodes_arr,'steps': self.episode_steps, 'total episode reward': self.episodes_reward})
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            df.to_csv(f'report/car_racing_{current_datetime}.csv', encoding='utf-8', index = False)
            self.label_comunication.place(x=70, y=320)
            self.label_comunication.config(text=f"car_racing_{current_datetime}.csv saved!", fg = "red")
        
        self.episode_number = 0
        
        #reset variables for csv
        self.episodes_reward = []
        self.episodes_arr = []
        self.episode_steps = []
    
    def button_home_clicked(self):

        self.tot_reward = 0
        self.check_field = False #check field setting, enable arrows 
        self.episode_number = 0
        self.episodes_arr = [] #store the number of episode
        self.episodes_reward = [] #save the tot_reward of every episode
        self.episode_steps = [] #save the number of steps of each episode
        self.terminated = False
        self.truncated = False
        self.steps = 0

        self.controller.show_frame("Home")