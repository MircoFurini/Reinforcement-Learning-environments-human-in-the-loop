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
import time
import pickle

class MountainCarPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#FFFFFF")
        self.controller = controller

        self.tot_reward = 0 

        #for the csv
        self.episode_number = 0
        self.episodes_arr = [] #store the number of episode
        self.episodes_reward = [] #save the tot_reward of every episode
        self.episode_steps = [] #save the number of steps of each episode
        self.episodes_type = []
        self.episode_type_string = "human"

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
            text="MOUNTAIN CAR DASHBOARD",
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
        self.label_comunication.place(x=40, y=200)

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
            x=620.0,
            y=110.0,
            width=240.0,
            height=70.0
        )

        #qlearning button
        self.button_qlearning = Button(
            self,
            text="QLEARNING ALGORITHM",
            font = ("Inter", 10, "bold"),
            bg="#F38630",
            fg="#FFFFFF",
            command=self.button_qlearning_clicked,
        )
        self.button_qlearning.place(
            x=330.0,
            y=110.0,
            width=240.0,
            height=70.0
        )

        #qlearning score button
        self.button_score = Button(
            self,
            text="QLEARNING SCORE",
            font = ("Inter", 10, "bold"),
            bg="#F38630",
            fg="#FFFFFF",
            command=self.button_score_clicked,
        )
        self.button_score.place(
            x=910.0,
            y=110.0,
            width=240.0,
            height=70.0
        )

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
        # Inizialization graph
        self.fig, self.axs = plt.subplots(2, 2, figsize=(11, 8))
        plt.subplots_adjust(hspace=0.5, wspace=0.3)

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
        self.ax3.set_ylim(-0.5,2.5)

        self.x_data_gas = []  
        self.y_data_gas = []  
        self.line_gas, = self.ax4.plot(self.x_data_gas, self.y_data_gas, 'go')

        # Configuration graph
        self.ax1.set_title('Total reward')
        self.ax1.set_xlabel('step')
        self.ax1.set_ylabel('total reward')

        self.ax2.set_title('Reward')
        self.ax2.set_xlabel('step')
        self.ax2.set_ylabel('reward')

        self.ax3.set_yticks([0,1,2])
        self.ax3.set_yticklabels(['left', 'noop', 'right'])
        self.ax3.set_title('Steer')
        self.ax3.set_xlabel('step')
        self.ax3.set_ylabel('steer')

        self.ax4.set_title('Velocity')
        self.ax4.set_xlabel('step')
        self.ax4.set_ylabel('velocity')

        self.ax2.set_xlim(-0.5,10)
        self.ax3.set_xlim(-0.5,10)
        self.ax4.set_xlim(-0.5,10)

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
        self.episode_type_string = "human"

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

        plt.draw()  
        
        #delete the preliminary widgets
        self.button_save.place_forget()
        self.button_new_episode.place_forget()
        self.button_qlearning.place_forget()
        self.button_score.place_forget()
        self.label_comunication.place(x=70, y=200)
        self.label_comunication.config(text = "Use \'ad\' keys \nto control the car. \n\nClose the game window \nto start a new episode", fg ="#8398E3")
        
        #start game
        play(gym.make('MountainCar-v0', autoreset = False, render_mode="rgb_array"),callback=self.callback, keys_to_action={
            "a": 0,
            "d": 2,
        }, noop=1)
        
        #place again the buttons after the game
        self.label_comunication.place_forget()
        
        self.button_new_episode.place(
            x=40.0,
            y=110.0,
            width=240.0,
            height=70.0
        )

        self.button_qlearning.place(
            x=330.0,
            y=110.0,
            width=240.0,
            height=70.0
        )

        self.button_save.place(
            x=620.0,
            y=110.0,
            width=240.0,
            height=70.0
            
        ) 

        self.button_score.place(
            x=910.0,
            y=110.0,
            width=240.0,
            height=70.0
        )

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
        
        self.update()

    #call by play function
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
        self.y_data_steer.append(action)
        self.y_data_steer= self.y_data_steer[-10:]  
        self.line_steer.set_ydata(self.y_data_steer)
        self.line_steer.set_xdata(self.x_data_steer)
        self.ax3.relim()  # Ridisegna i limiti dell'asse
        self.ax3.autoscale_view(True, True, True)  # Ridisegna l'asse in base ai nuovi dati
        self.ax3.set_xlabel(f'Step: {self.steps}')

        self.x_data_gas.append(self.steps)
        self.x_data_gas= self.x_data_gas[-10:]  
        self.y_data_gas.append(obs_tp1[1])
        self.y_data_gas= self.y_data_gas[-10:]  
        self.line_gas.set_ydata(self.y_data_gas)
        self.line_gas.set_xdata(self.x_data_gas)

        
        self.ax4.relim()  # Ridisegna i limiti dell'asse
        self.ax4.autoscale_view(True, True, True)  # Ridisegna l'asse in base ai nuovi dati
        self.ax4.set_xlabel(f'Step: {self.steps}')

        if self.steps <= 9:
            self.ax2.set_xlim(-0.5, 10)
            self.ax3.set_xlim(-0.5, 10)
            self.ax4.set_xlim(-0.5, 10)
        
        else:
            self.ax2.set_xlim(self.steps -10, self.steps+1)
            self.ax3.set_xlim(self.steps -10, self.steps+1)
            self.ax4.set_xlim(self.steps -10, self.steps+1)
        
        plt.draw()
        self.update()

        #Manage the episode 
        if terminated or truncated:
            
            self.episodes_reward.append(self.tot_reward)
            self.episodes_arr.append(self.episode_number)
            self.episode_steps.append(self.steps)
            self.episodes_type.append(self.episode_type_string)
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

            plt.draw()

        self.steps+=1
    
    #button save 
    def button_save_clicked(self): #save data and restart episode
            
        if self.episodes_arr == []:
            self.label_comunication.place(x=70, y=200)
            self.label_comunication.config(text=f"No data to save! Restart a match!", fg = "red")
        else:
            df = pd.DataFrame({'episode': self.episodes_arr,'steps': self.episode_steps, 'total episode reward': self.episodes_reward, 'type': self.episodes_type})
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            df.to_csv(f'report/mountain_car_{current_datetime}.csv', encoding='utf-8', index = False)
            self.label_comunication.place(x=70, y=200)
            self.label_comunication.config(text=f"mountain_car_{current_datetime}.csv saved!", fg = "red")
        
        self.episode_number = 0
        
        #reset variables for csv
        self.episodes_reward = []
        self.episodes_arr = []
        self.episode_steps = []
        self.episodes_type = []
    

    ####functions for qlearning
        
    def button_qlearning_clicked(self):
        
        self.episode_type_string = "algorithm"

        mountain_car = gym.make('MountainCar-v0', autoreset = False, render_mode="human")

        upperBounds=mountain_car.observation_space.high
        lowerBounds=mountain_car.observation_space.low

        #hide save and new episode bottons
        self.button_qlearning.place_forget()
        self.button_new_episode.place_forget()
        self.button_save.place_forget()
        self.button_score.place_forget()

        #open policy matrix
        with open('code/model/q_matrix.pkl', 'rb') as file:
            loaded_q_matrix = pickle.load(file)

        timeSteps=1000

        (currentState,_)=mountain_car.reset()

        for timeIndex in range(timeSteps):
            print(timeIndex)
            # select greedy actions
            actionInStateS=np.random.choice(np.where(loaded_q_matrix[self.returnIndexState(currentState, upperBounds, lowerBounds)]==np.max(loaded_q_matrix[self.returnIndexState(currentState, upperBounds, lowerBounds)]))[0])
            currentState, reward, terminated, truncated, info = mountain_car.step(actionInStateS)
            
            self.tot_reward+=reward

            ####update graph
            self.x_data_tot_reward.append(self.steps)
            self.y_data_tot_reward.append(self.tot_reward)
            self.line_tot_reward.set_ydata(self.y_data_tot_reward)
            self.line_tot_reward.set_xdata(self.x_data_tot_reward)
            self.ax1.relim()  # Ridisegna i limiti dell'asse
            self.ax1.autoscale_view(True, True, True)  # Ridisegna l'asse in base ai nuovi dati
            self.ax1.set_xlabel(f'Step: {self.steps}')

            self.x_data_rew.append(self.steps)
            self.x_data_rew = self.x_data_rew[-10:]  # mantieni solo gli ultimi 100 punti
            self.y_data_rew.append(reward)
            self.y_data_rew = self.y_data_rew[-10:]  # mantieni solo gli ultimi 100 punti
            self.line_rew.set_ydata(self.y_data_rew)
            self.line_rew.set_xdata(self.x_data_rew)
            self.ax2.relim()  # Ridisegna i limiti dell'asse
            self.ax2.autoscale_view(True, True, True)  # Ridisegna l'asse in base ai nuovi dati
            self.ax2.set_xlabel(f'Step: {self.steps}')

            self.x_data_steer.append(self.steps)
            self.x_data_steer= self.x_data_steer[-10:]  
            self.y_data_steer.append(actionInStateS)
            self.y_data_steer= self.y_data_steer[-10:]  
            self.line_steer.set_ydata(self.y_data_steer)
            self.line_steer.set_xdata(self.x_data_steer)
            self.ax3.relim()  # Ridisegna i limiti dell'asse
            self.ax3.autoscale_view(True, True, True)  # Ridisegna l'asse in base ai nuovi dati
            self.ax3.set_xlabel(f'Step: {self.steps}')

            self.x_data_gas.append(self.steps)
            self.x_data_gas= self.x_data_gas[-10:]  
            self.y_data_gas.append(currentState[1])
            self.y_data_gas= self.y_data_gas[-10:]  
            self.line_gas.set_ydata(self.y_data_gas)
            self.line_gas.set_xdata(self.x_data_gas)

            self.ax4.relim()  # Ridisegna i limiti dell'asse
            self.ax4.autoscale_view(True, True, True)  # Ridisegna l'asse in base ai nuovi dati
            self.ax4.set_xlabel(f'Step: {self.steps}')
            
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
            ######

            self.steps+=1

            time.sleep(0.05)
            if terminated or truncated:
            
                self.episodes_reward.append(self.tot_reward)
                self.episodes_arr.append(self.episode_number)
                self.episode_steps.append(self.steps)
                self.episodes_type.append(self.episode_type_string)
                print(f"Episode tot_reward: {self.tot_reward}")
                self.episode_number+=1
                self.tot_reward = 0
                self.steps = 0

            if (terminated):
                time.sleep(1)
                break
        
        self.steps = 0
        self.tot_reward = 0

        #replace buttons
        self.button_new_episode.place(
            x=40.0,
            y=110.0,
            width=240.0,
            height=70.0
        )

        self.button_qlearning.place(
            x=330.0,
            y=110.0,
            width=240.0,
            height=70.0
        )

        self.button_save.place(
            x=620.0,
            y=110.0,
            width=240.0,
            height=70.0
            
        ) 

        self.button_score.place(
            x=910.0,
            y=110.0,
            width=240.0,
            height=70.0
        )

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
        
        plt.draw()
        self.update()

        mountain_car.close()
    
    #for qlearning
    def returnIndexState(self,state, upper_Bounds, lower_Bounds):

        #variables for discretization
        upperBounds=upper_Bounds
        lowerBounds=lower_Bounds
        positionMin=-1.2
        positionMax=0.6
        velocityMin=-0.07
        velocityMax=0.07
        upperBounds[0]=positionMax
        upperBounds[1]=velocityMax
        lowerBounds[0]=positionMin
        lowerBounds[1]=velocityMin

        numberOfBinsPosition=30
        numberOfBinsVelocity=30
        numberOfBins=[numberOfBinsPosition,numberOfBinsVelocity]
        
        cartPositionBin=np.linspace(lowerBounds[0],upperBounds[0],numberOfBins[0])
        cartVelocityBin=np.linspace(lowerBounds[1],upperBounds[1],numberOfBins[1])
        
        indexPosition=np.maximum(np.digitize(state[0],cartPositionBin)-1,0)
        indexVelocity=np.maximum(np.digitize(state[1],cartVelocityBin)-1,0)
        
        return tuple([indexPosition,indexVelocity])

    def button_score_clicked(self):

        mountain_car = gym.make('MountainCar-v0', autoreset = False)

        upperBounds=mountain_car.observation_space.high
        lowerBounds=mountain_car.observation_space.low

        #hide save and new episode bottons
        self.button_new_episode.place_forget()
        self.button_save.place_forget()
        self.button_score.place_forget()
        self.button_qlearning.place_forget()

        #open policy matrix
        with open('code/model/q_matrix.pkl', 'rb') as file:
            loaded_q_matrix = pickle.load(file)

        (currentState,_)=mountain_car.reset()

        total_reward = 0 #single episode reward
        total_episodes_reward = 0 #cumulative reward of episodes
        terminated = False
        truncated = False
        episode_number = 0

        while episode_number != 10:
            # select greedy actions
            actionInStateS=np.random.choice(np.where(loaded_q_matrix[self.returnIndexState(currentState, upperBounds, lowerBounds)]==np.max(loaded_q_matrix[self.returnIndexState(currentState, upperBounds, lowerBounds)]))[0])
            currentState, reward, terminated, truncated, info = mountain_car.step(actionInStateS)
            
            total_reward+=reward

            if terminated or truncated:
                print(f"Episode number:{episode_number}, total reward: {total_reward}")
                
                total_episodes_reward += total_reward
                mountain_car.reset()

                total_reward = 0
                episode_number+=1

        average_reward = total_episodes_reward/10
        
        self.label_comunication.place(x=70, y=200)
        self.label_comunication.config(text = f"The average score is:{average_reward}")

        self.update

        mountain_car.close()

        #replace buttons
        self.button_new_episode.place(
            x=40.0,
            y=110.0,
            width=240.0,
            height=70.0
        )

        self.button_qlearning.place(
            x=330.0,
            y=110.0,
            width=240.0,
            height=70.0
        )

        self.button_save.place(
            x=620.0,
            y=110.0,
            width=240.0,
            height=70.0
            
        ) 

        self.button_score.place(
            x=910.0,
            y=110.0,
            width=240.0,
            height=70.0
        )

    def button_home_clicked(self):
        
        #reset the variables
        self.tot_reward = 0
        self.episode_number = 0
        self.episodes_arr = [] #store the number of episode
        self.episodes_reward = [] #save the tot_reward of every episode
        self.episode_steps = [] #save the number of steps of each episode
        self.episodes_type = []
        self.terminated = False
        self.truncated = False
        self.steps = 0

        ##Reset graph 
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

        plt.draw() 

        self.controller.show_frame("Home")