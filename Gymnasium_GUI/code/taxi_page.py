
import gymnasium as gym 
from collections import defaultdict
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from PIL import Image,ImageTk
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
from pathlib import Path
import pygame
import numpy as np
import importlib
from model.taxi import Taxi
from datetime import datetime
import time
from copy import deepcopy

class TaxiPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#FFFFFF")
        self.controller = controller
        
        
        self.check_field = False #check field setting, enable arrows 

        self.episode_number = 0
        self.episodes_arr = [] #store the number of episode
        self.episodes_reward = [] #save the tot_reward of every episode
        self.episode_steps = [] #save the number of steps of each episode
        self.episodes_type = [] #save episodes type (human, suggestions, algorithm)
        self.episode_type_string = "human" #save the episode type temporarily
        
        self.tot_reward = 0
        self.terminated = False
        self.truncated = False

        self.image_render = None
        self.check_buttons_suggestions_clicked = False

        self.environment_started = False #for home button
        #######################title canvas

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
            text="TAXI DASHBOARD",
            fill="#FFFFFF",
            font=("Inter Bold", 40 * -1)
        )

        #label for comunications ex. save button
        self.label_comunication = Label(self,
                           text="",
                           font=("Inter", 15, "bold"),
                           fg="#8398E3",
                           bg="#FFFFFF")
        self.label_comunication.place(x=43, y=590)

        ######## GAME BUTTONS
        
        #button start
        self.button_start = Button(
            self,
            text="START",
            font = ("Inter", 10, "bold"),
            bg="#F38630",
            fg="#FFFFFF",
            command=self.button_start_clicked,
        )
        self.button_start.place(
            x=145.0,
            y=454.0,
            width=240.0,
            height=70.0
        )

        self.button_new_episode = Button(
            self,
            text="NEW EPISODE",
            font = ("Inter", 10, "bold"),
            bg="#F38630",
            fg="#FFFFFF",
            command=self.button_new_episode_clicked,
        )

        #button new episode suggestions
        self.button_episode_suggestions = Button(
            self,
            text="NEW EPISODE WITH SUGGESTIONS",
            font = ("Inter", 10, "bold"),
            bg="#F38630",
            fg="#FFFFFF",
            command=self.button_episode_suggestions_clicked,
        )

        #qLearning button
        self.button_qlearning = Button(
            self,
            text="QLearning EPISODE",
            font = ("Inter", 10, "bold"),
            bg="#F38630",
            fg="#FFFFFF",
            command=self.button_qlearning_clicked,
        )

        #button save
        self.button_save = Button(
            self,
            text="SAVE EPISODE REPORT",
            font = ("Inter", 10, "bold"),
            bg="#F38630",
            fg="#FFFFFF",
            command=self.button_save_clicked,
        )


        ########GAME BUTTONS
        
        self.button_up = Button(
            self,
            text="▲",
            font = ("Inter", 20, "bold"),
            bg="#F38630",
            fg="#FFFFFF",
            command=self.button_up_clicked,
        )

        self.button_right = Button(
            self,
            text="►",
            font = ("Inter", 20, "bold"),
            bg="#F38630",
            fg="#FFFFFF",
            command=self.button_right_clicked,
        )

        self.button_down = Button(
            self,
            text="▼",
            font = ("Inter", 20, "bold"),
            bg="#F38630",
            fg="#FFFFFF",
            command=self.button_down_clicked,
        )

        self.button_left = Button(
            self,
            text="◄",
            font = ("Inter", 20, "bold"),
            bg="#F38630",
            fg="#FFFFFF",
            command=self.button_left_clicked,
        )

        self.button_pickup = Button(
            self,
            text="PICKUP",
            font = ("Inter", 20, "bold"),
            bg="#F38630",
            fg="#FFFFFF",
            command=self.button_pickup_clicked,
        )

        self.button_dropoff = Button(
            self,
            text="DROPOFF",
            font = ("Inter", 20, "bold"),
            bg="#F38630",
            fg="#FFFFFF",
            command=self.button_dropoff_clicked,
        )

        button_home = Button(
            self,
            text="HOME",
            font = ("Inter", 10, "bold"),
            bg="#F38630",
            fg="#FFFFFF",
            command=self.button_home_clicked,
        )
        button_home.place(
            x=43.0,
            y=880.0,
            width=240.0,
            height=70.0
        )

        self.button_score = Button(
            self,
            text="Q-LEARNING SCORE",
            font = ("Inter", 10, "bold"),
            bg="#F38630",
            fg="#FFFFFF",
            command=self.button_score_clicked,
        )

        ############graph
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(11, 8))
        plt.subplots_adjust(hspace=0.7)

        self.x_data_tot_reward = []  
        self.y_data_tot_reward = []  
        self.line_tot_reward, = self.ax1.plot(self.x_data_tot_reward, self.y_data_tot_reward, 'ro-')
        self.x_data_tot_reward_predict = []  
        self.y_data_tot_reward_predict = []  
        self.line_tot_reward_predict, = self.ax1.plot(self.x_data_tot_reward_predict, self.y_data_tot_reward_predict, 'go')

        self.x_data_rew = []  
        self.y_data_rew = [] 
        self.line_rew, = self.ax2.plot(self.x_data_rew, self.y_data_rew, 'ro-')
        self.x_data_rew_predict = []  
        self.y_data_rew_predict = [] 
        self.line_rew_predict, = self.ax2.plot(self.x_data_rew_predict, self.y_data_rew_predict, 'go')

        self.x_data_steer = []  
        self.y_data_steer = []  
        self.line_steer, = self.ax3.plot(self.x_data_steer, self.y_data_steer, 'ro')
        self.ax3.set_ylim(-0.5,5.5)
        self.x_data_steer_predict = []  
        self.y_data_steer_predict = []  
        self.line_steer_predict, = self.ax3.plot(self.x_data_steer_predict, self.y_data_steer_predict, 'go')

        # Configuration tot_rewards, reward, actions
        self.ax1.set_title('Total reward')
        self.ax1.set_xlabel('step')
        self.ax1.set_ylabel('total reward')

        self.ax2.set_title('Reward')
        self.ax2.set_xlabel('step')
        self.ax2.set_ylabel('reward')

        self.ax3.set_yticks([0, 1, 2, 3, 4, 5])
        self.ax3.set_yticklabels(['Down','Up','Right','Left','Pickup','Drop off'])
        self.ax3.set_title('Action')
        self.ax3.set_xlabel('step')
        self.ax3.set_ylabel('Actions')

        self.ax2.set_xlim(-0.5,10)
        self.ax3.set_xlim(-0.5,10)
        ###########

        ####graph visualization
        self.left_frame = tk.Frame(self, width=800, height=650)
        self.left_frame.place(x=800, y=200)
            
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0)

        self.update
        ##########


    ################ FUNCTIONS #################
        
    #button save & new episode
    
    def button_start_clicked(self):

        self.environment_started = True

        self.taxi = Taxi()
        self.button_start.place_forget()
        #display new buttons
        self.button_new_episode.place(
            x=40.0,
            y=110.0,
            width=240.0,
            height=70.0
        )

        self.button_episode_suggestions.place(
            x=330.0,
            y=110.0,
            width=240.0,
            height=70.0
        )

        self.button_qlearning.place(
            x=620.0,
            y=110.0,
            width=240.0,
            height=70.0
        )

        self.button_save.place(
            x=910.0,
            y=110.0,
            width=240.0,
            height=70.0
        ) 

        self.button_score.place(
            x=1200.0,
            y=110.0,
            width=240.0,
            height=70.0
        )
        

    def button_new_episode_clicked(self):
        
        self.label_comunication.config(text = "")

        self.check_buttons_suggestions_clicked = False

        #in order to allow the keys
        self.tot_reward = 0
        self.terminated = False
        self.truncated = False

        self.button_up.config(bg="#F38630")
        self.button_down.config(bg="#F38630")
        self.button_right.config(bg="#F38630")
        self.button_left.config(bg="#F38630")
        self.button_pickup.config(bg="#F38630")
        self.button_dropoff.config(bg="#F38630")

        self.taxi = Taxi()
        self.episode_type_string = "human"

        self.taxi_predict = deepcopy(self.taxi)

        #reset graph
        self.x_data_tot_reward = []  
        self.y_data_tot_reward = []
        self.line_tot_reward.set_ydata(self.y_data_tot_reward)
        self.line_tot_reward.set_xdata(self.x_data_tot_reward)  
        
        self.x_data_rew = []  
        self.y_data_rew = []
        self.line_rew.set_ydata(self.y_data_rew)
        self.line_rew.set_xdata(self.x_data_rew)  
        
        self.x_data_steer = []  
        self.y_data_steer = []
        self.line_steer.set_ydata(self.y_data_steer)
        self.line_steer.set_xdata(self.x_data_steer)  

        self.x_data_tot_reward_predict = []  
        self.y_data_tot_reward_predict = []  
        self.line_tot_reward_predict.set_ydata(self.y_data_tot_reward_predict)
        self.line_tot_reward_predict.set_xdata(self.x_data_tot_reward_predict)

        self.x_data_rew_predict = []  
        self.y_data_rew_predict = []  
        self.line_rew_predict.set_ydata(self.y_data_rew_predict)
        self.line_rew_predict.set_xdata(self.x_data_rew_predict)

        self.x_data_steer_predict = []  
        self.y_data_steer_predict = []
        self.line_steer_predict.set_ydata(self.y_data_steer_predict)
        self.line_steer_predict.set_xdata(self.x_data_steer_predict)

        plt.draw()

        image_load = self.taxi.render_image()
        self.image_render = ImageTk.PhotoImage(image_load, master=self)
        self.label_1 = tk.Label(self, image=self.image_render)
        self.label_1["background"] = "white"
        self.label_1.place(x=50, y=200)
        
        #reset buttons
        self.button_up.place(
            x=250.0,
            y=624.0,
            width=100.0,
            height=100.0
        )

        self.button_right.place(
            x=382.0,
            y=756.0,
            width=100.0,
            height=100.0
        )

        self.button_down.place(
            x=250.0,
            y=756.0,
            width=100.0,
            height=100.0
        )

        self.button_left.place(
            x=118.0,
            y=756.0,
            width=100.0,
            height=100.0
        )

        self.button_pickup.place(
            x=514.0,
            y=624.0,
            width=150.0,
            height=100.0
        )

        self.button_dropoff.place(
            x=514.0,
            y=756.0,
            width=150.0,
            height=100.0
        )

    def button_episode_suggestions_clicked(self):
        
        self.button_up.config(bg="#F38630")
        self.button_down.config(bg="#F38630")
        self.button_right.config(bg="#F38630")
        self.button_left.config(bg="#F38630")
        self.button_pickup.config(bg="#F38630")
        self.button_dropoff.config(bg="#F38630")

        self.button_new_episode_clicked()

        self.episode_type_string = "suggestions"

        self.check_buttons_suggestions_clicked = True

        observation = int(self.taxi.get_state())
        self.action_predict = self.taxi.learnedAction(observation)
        print(f"ACTION PREDICT: {self.action_predict}")

        if self.action_predict == 1:
            self.button_up.config(bg="#30F38E")
        if self.action_predict == 0:
            self.button_down.config(bg="#30F38E")
        if self.action_predict == 2:
            self.button_right.config(bg="#30F38E")
        if self.action_predict == 3:
            self.button_left.config(bg="#30F38E")
        if self.action_predict == 4:
            self.button_pickup.config(bg="#30F38E")
        if self.action_predict == 5:
            self.button_dropoff.config(bg="#30F38E")

    def button_qlearning_clicked(self): 
        
        self.label_comunication.config(text = "")

        self.check_buttons_suggestions_clicked = False

        self.button_up.config(bg="#F38630")
        self.button_down.config(bg="#F38630")
        self.button_right.config(bg="#F38630")
        self.button_left.config(bg="#F38630")
        self.button_pickup.config(bg="#F38630")
        self.button_dropoff.config(bg="#F38630")

        self.episode_type_string = "algorithm"

        self.tot_reward = 0
        self.truncated = False
        self.terminated = False

        self.taxi.restart_env()

        #reset graph
        self.x_data_tot_reward = []  
        self.y_data_tot_reward = []
        self.line_tot_reward.set_ydata(self.y_data_tot_reward)
        self.line_tot_reward.set_xdata(self.x_data_tot_reward)  
        
        self.x_data_rew = []  
        self.y_data_rew = []
        self.line_rew.set_ydata(self.y_data_rew)
        self.line_rew.set_xdata(self.x_data_rew)  
        
        self.x_data_steer = []  
        self.y_data_steer = []
        self.line_steer.set_ydata(self.y_data_steer)
        self.line_steer.set_xdata(self.x_data_steer)  

        self.x_data_tot_reward_predict = []  
        self.y_data_tot_reward_predict = []  
        self.line_tot_reward_predict.set_ydata(self.y_data_tot_reward_predict)
        self.line_tot_reward_predict.set_xdata(self.x_data_tot_reward_predict)

        self.x_data_rew_predict = []  
        self.y_data_rew_predict = []  
        self.line_rew_predict.set_ydata(self.y_data_rew_predict)
        self.line_rew_predict.set_xdata(self.x_data_rew_predict)

        self.x_data_steer_predict = []  
        self.y_data_steer_predict = []
        self.line_steer_predict.set_ydata(self.y_data_steer_predict)
        self.line_steer_predict.set_xdata(self.x_data_steer_predict)
    
        self.ax2.set_xlim(-0.5,10)
        self.ax3.set_xlim(-0.5,10)

        plt.draw()    

        self.button_episode_suggestions.place_forget()
        self.button_new_episode.place_forget()
        self.button_save.place_forget()
        self.button_qlearning.place_forget()
        self.button_score.place_forget()

        self.button_up.place_forget()
        self.button_down.place_forget()
        self.button_left.place_forget()
        self.button_right.place_forget()
        self.button_pickup.place_forget()
        self.button_dropoff.place_forget()

        image_load = self.taxi.render_image()
        self.image_render = ImageTk.PhotoImage(image_load, master=self)
        self.label_1 = tk.Label(self, image=self.image_render)
        self.label_1["background"] = "white"
        self.label_1.place(x=50, y=200)
        
        while not self.terminated and not self.truncated:

            observation = int(self.taxi.get_state())
            action = self.taxi.learnedAction(observation)
            
            print(f"State:{observation}, action:{action}")
            print(action)
            reward, steps, self.terminated, self.truncated, observation = self.taxi.select_action(action)

            image_load = self.taxi.render_image()
            self.image_render = ImageTk.PhotoImage(image_load, master=self)
            self.label_1.config(image=self.image_render)

            self.update_graph(reward, steps, action, observation)
            
            self.update()

            time.sleep(1) #to see the render
        
        self.episodes_reward.append(self.tot_reward)
        self.episodes_arr.append(self.episode_number)
        steps = self.taxi.get_steps()
        self.episode_steps.append(steps)
        self.episodes_type.append(self.episode_type_string)
        print(f"Episode tot_reward: {self.tot_reward}")

        self.episode_number+=1
        self.tot_reward = 0 

        self.label_comunication.config(text = "Episode terminated, please restart a match!")

        self.button_new_episode.place(
            x=40.0,
            y=110.0,
            width=240.0,
            height=70.0
        )

        self.button_episode_suggestions.place(
            x=330.0,
            y=110.0,
            width=240.0,
            height=70.0
        )

        self.button_qlearning.place(
            x=620.0,
            y=110.0,
            width=240.0,
            height=70.0
        )

        self.button_save.place(
            x=910.0,
            y=110.0,
            width=240.0,
            height=70.0
        ) 

        self.button_score.place(
            x=1200.0,
            y=110.0,
            width=240.0,
            height=70.0
        )
    
    def button_save_clicked(self): #save data and restart episode

        self.label_comunication.config(text = "")

        self.button_up.config(bg="#F38630")
        self.button_down.config(bg="#F38630")
        self.button_right.config(bg="#F38630")
        self.button_left.config(bg="#F38630")
        self.button_pickup.config(bg="#F38630")
        self.button_dropoff.config(bg="#F38630")

        if self.episodes_arr == []:
            self.label_comunication.place(x=43, y=590)
            self.label_comunication.config(text=f"No data to save! Restart a match!", fg = "red")
        else:    
            df = pd.DataFrame({'episode': self.episodes_arr,'steps': self.episode_steps, 'total episode reward': self.episodes_reward, 'type': self.episodes_type})
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            df.to_csv(f'report/taxi_{current_datetime}.csv', encoding='utf-8', index = False)
            self.label_comunication.config(text=f"taxi_{current_datetime}.csv saved!", fg = "red")

        #reset variables
        print(f"Episode tot_reward: {self.tot_reward}")
        self.tot_reward = 0
        self.episode_number = 0

        #reset variables for csv
        self.episodes_reward = []
        self.episodes_arr = []
        self.episode_steps = []
        self.episodes_type = []

        #restart env
        self.taxi.restart_env()

        #white image
        self.label_1 = tk.Label(self)
        self.label_1["background"] = "white"
        self.label_1.place(x=50, y=200)

        #reset graph
        self.x_data_tot_reward = []  
        self.y_data_tot_reward = []
        self.line_tot_reward.set_ydata(self.y_data_tot_reward)
        self.line_tot_reward.set_xdata(self.x_data_tot_reward)  
        
        self.x_data_rew = []  
        self.y_data_rew = []
        self.line_rew.set_ydata(self.y_data_rew)
        self.line_rew.set_xdata(self.x_data_rew)  
        
        self.x_data_steer = []  
        self.y_data_steer = []
        self.line_steer.set_ydata(self.y_data_steer)
        self.line_steer.set_xdata(self.x_data_steer)  

        self.x_data_tot_reward_predict = []  
        self.y_data_tot_reward_predict = []  
        self.line_tot_reward_predict.set_ydata(self.y_data_tot_reward_predict)
        self.line_tot_reward_predict.set_xdata(self.x_data_tot_reward_predict)

        self.x_data_rew_predict = []  
        self.y_data_rew_predict = []  
        self.line_rew_predict.set_ydata(self.y_data_rew_predict)
        self.line_rew_predict.set_xdata(self.x_data_rew_predict)

        self.x_data_steer_predict = []  
        self.y_data_steer_predict = []
        self.line_steer_predict.set_ydata(self.y_data_steer_predict)
        self.line_steer_predict.set_xdata(self.x_data_steer_predict)

        self.ax2.set_xlim(-0.5,10)
        self.ax3.set_xlim(-0.5,10)

        plt.draw()

        self.check_buttons_suggestions_clicked = False

    def button_score_clicked(self):
        self.taxi.restart_env()
        
        total_rew = 0
        terminated = False
        truncated = False

        for _ in range(0,10):
            self.taxi.restart_env()
            terminated = False
            truncated = False
            while not terminated and not truncated:
                observation = int(self.taxi.get_state())
                action = self.taxi.learnedAction(observation)
                reward, steps, terminated, truncated, observation = self.taxi.select_action(action)

                total_rew+=reward
            print(f"Total reward:{total_rew}")

        self.label_comunication.config(text = f"Q-learning average score is {total_rew/10}. Please, restart a match!")

        self.taxi.restart_env()

        #reset graph
        self.x_data_tot_reward = []  
        self.y_data_tot_reward = []
        self.line_tot_reward.set_ydata(self.y_data_tot_reward)
        self.line_tot_reward.set_xdata(self.x_data_tot_reward)  
        
        self.x_data_rew = []  
        self.y_data_rew = []
        self.line_rew.set_ydata(self.y_data_rew)
        self.line_rew.set_xdata(self.x_data_rew)  
        
        self.x_data_steer = []  
        self.y_data_steer = []
        self.line_steer.set_ydata(self.y_data_steer)
        self.line_steer.set_xdata(self.x_data_steer)  

        self.x_data_tot_reward_predict = []  
        self.y_data_tot_reward_predict = []  
        self.line_tot_reward_predict.set_ydata(self.y_data_tot_reward_predict)
        self.line_tot_reward_predict.set_xdata(self.x_data_tot_reward_predict)

        self.x_data_rew_predict = []  
        self.y_data_rew_predict = []  
        self.line_rew_predict.set_ydata(self.y_data_rew_predict)
        self.line_rew_predict.set_xdata(self.x_data_rew_predict)

        self.x_data_steer_predict = []  
        self.y_data_steer_predict = []
        self.line_steer_predict.set_ydata(self.y_data_steer_predict)
        self.line_steer_predict.set_xdata(self.x_data_steer_predict)

        plt.draw()

        image_load = self.taxi.render_image()
        self.image_render = ImageTk.PhotoImage(image_load, master=self)
        self.label_1 = tk.Label(self, image=self.image_render)
        self.label_1["background"] = "white"
        self.label_1.place(x=50, y=200)

        self.tot_reward = 0
        self.check_buttons_suggestions_clicked = False

        self.terminated = True
        self.truncated = True 

    def update_graph(self, reward, steps, action, observation):   
        self.tot_reward+=reward
    
        self.x_data_tot_reward.append(steps)
        self.y_data_tot_reward.append(self.tot_reward)
        self.line_tot_reward.set_ydata(self.y_data_tot_reward)
        self.line_tot_reward.set_xdata(self.x_data_tot_reward)
        
        self.x_data_rew.append(steps)
        self.x_data_rew = self.x_data_rew[-10:]  # Mantieni solo gli ultimi 10 punti
        self.y_data_rew.append(reward)
        self.y_data_rew = self.y_data_rew[-10:]  # Mantieni solo gli ultimi 10 punti
        self.line_rew.set_ydata(self.y_data_rew)
        self.line_rew.set_xdata(self.x_data_rew)
        
        self.x_data_steer.append(steps)
        self.x_data_steer= self.x_data_steer[-10:]  
        self.y_data_steer.append(action)
        self.y_data_steer= self.y_data_steer[-10:]  
        self.line_steer.set_ydata(self.y_data_steer)
        self.line_steer.set_xdata(self.x_data_steer)
        

        if self.check_buttons_suggestions_clicked:
            self.x_data_tot_reward_predict.append(steps)
            self.x_data_tot_reward_predict = self.x_data_tot_reward_predict[-1:]
            self.y_data_tot_reward_predict.append(self.tot_reward_predict)
            self.y_data_tot_reward_predict = self.y_data_tot_reward_predict[-1:]
            self.line_tot_reward_predict.set_ydata(self.y_data_tot_reward_predict)
            self.line_tot_reward_predict.set_xdata(self.x_data_tot_reward_predict)

            self.x_data_rew_predict.append(steps)
            self.x_data_rew_predict = self.x_data_rew_predict[-1:]  # Mantieni solo gli ultimi 10 punti
            self.y_data_rew_predict.append(self.reward_predict)
            self.y_data_rew_predict = self.y_data_rew_predict[-1:]  # Mantieni solo gli ultimi 10 punti
            self.line_rew_predict.set_ydata(self.y_data_rew_predict)
            self.line_rew_predict.set_xdata(self.x_data_rew_predict)

            self.x_data_steer_predict.append(steps)
            self.x_data_steer_predict= self.x_data_steer_predict[-1:]  
            self.y_data_steer_predict.append(self.action_predict)
            self.y_data_steer_predict= self.y_data_steer_predict[-1:]  
            self.line_steer_predict.set_ydata(self.y_data_steer_predict)
            self.line_steer_predict.set_xdata(self.x_data_steer_predict)

        self.ax1.relim()  # Ridisegna i limiti dell'asse
        self.ax1.autoscale_view(True, True, True)  # Ridisegna l'asse in base ai nuovi dati
        self.ax1.set_xlabel(f'Step: {steps}, state: {observation}, total reward: {self.tot_reward}')# Update x-axis label with current step number

        self.ax2.relim()  # Ridisegna i limiti dell'asse
        self.ax2.autoscale_view(True, True, True)  # Ridisegna l'asse in base ai nuovi dati
        self.ax2.set_xlabel(f'Step: {steps}, state: {observation}, reward: {reward}')

        self.ax3.relim()  # Ridisegna i limiti dell'asse
        self.ax3.autoscale_view(True, True, True)  # Ridisegna l'asse in base ai nuovi dati
        self.ax3.set_xlabel(f'Step: {steps}, Action: {action}')

        if steps <= 9:
            self.ax2.set_xlim(-0.5, 10)
            self.ax3.set_xlim(-0.5, 10)
        
        else:
            self.ax2.set_xlim(steps -10, steps+1)
            self.ax3.set_xlim(steps -10, steps+1)
        

        plt.draw() 
        self.canvas.draw() 
        self.update()
        
    ###################buttons actions
    def button_up_clicked(self):
        
        if not self.terminated and not self.truncated:
            if self.check_buttons_suggestions_clicked:
                taxi_predict_2 = deepcopy(self.taxi_predict)
                self.reward_predict, steps_predict, terminated_predict, truncated_predict, observation_predict = taxi_predict_2.select_action(self.action_predict)

                self.tot_reward_predict = self.tot_reward + self.reward_predict 

                self.taxi_predict.select_action(1)

            reward, steps, self.terminated, self.truncated, observation = self.taxi.select_action(1)
            image_load = self.taxi.render_image()
            self.image_render = ImageTk.PhotoImage(image_load, master=self)
            self.label_1.config(image=self.image_render)
            self.update_graph(reward, steps, 1, observation)
            
            if self.check_buttons_suggestions_clicked:
                self.button_up.config(bg="#F38630")
                self.button_down.config(bg="#F38630")
                self.button_right.config(bg="#F38630")
                self.button_left.config(bg="#F38630")
                self.button_pickup.config(bg="#F38630")
                self.button_dropoff.config(bg="#F38630")

                observation = int(self.taxi.get_state())
                self.action_predict = self.taxi.learnedAction(observation)

                if self.action_predict == 1:
                    self.button_up.config(bg="#30F38E")
                if self.action_predict == 0:
                    self.button_down.config(bg="#30F38E")
                if self.action_predict == 2:
                    self.button_right.config(bg="#30F38E")
                if self.action_predict == 3:
                    self.button_left.config(bg="#30F38E")
                if self.action_predict == 4:
                    self.button_pickup.config(bg="#30F38E")
                if self.action_predict == 5:
                    self.button_dropoff.config(bg="#30F38E")
            
            if self.terminated or self.truncated:
                self.episodes_reward.append(self.tot_reward)
                self.episodes_arr.append(self.episode_number)
                steps = self.taxi.get_steps()
                self.episode_steps.append(steps)
                self.episodes_type.append(self.episode_type_string)
                print(f"Episode tot_reward: {self.tot_reward}")

                self.episode_number+=1
                self.tot_reward = 0
                self.check_buttons_suggestions_clicked = False 

                self.label_comunication.config(text = "Episode terminated, please restart a match!")
        
    def button_down_clicked(self):
        if not self.terminated and not self.truncated:
            if self.check_buttons_suggestions_clicked:
                taxi_predict_2 = deepcopy(self.taxi_predict)
                self.reward_predict, steps_predict, terminated_predict, truncated_predict, observation_predict = taxi_predict_2.select_action(self.action_predict)

                self.tot_reward_predict = self.tot_reward + self.reward_predict

                self.taxi_predict.select_action(0)

            reward, steps, self.terminated, self.truncated, observation = self.taxi.select_action(0)
            image_load = self.taxi.render_image()
            self.image_render = ImageTk.PhotoImage(image_load, master=self)
            self.label_1.config(image=self.image_render)
            self.update_graph(reward, steps, 0, observation)

            if self.check_buttons_suggestions_clicked:
                self.button_up.config(bg="#F38630")
                self.button_down.config(bg="#F38630")
                self.button_right.config(bg="#F38630")
                self.button_left.config(bg="#F38630")
                self.button_pickup.config(bg="#F38630")
                self.button_dropoff.config(bg="#F38630")

                observation = int(self.taxi.get_state())
                self.action_predict = self.taxi.learnedAction(observation)

                if self.action_predict == 1:
                    self.button_up.config(bg="#30F38E")
                if self.action_predict == 0:
                    self.button_down.config(bg="#30F38E")
                if self.action_predict == 2:
                    self.button_right.config(bg="#30F38E")
                if self.action_predict == 3:
                    self.button_left.config(bg="#30F38E")
                if self.action_predict == 4:
                    self.button_pickup.config(bg="#30F38E")
                if self.action_predict == 5:
                    self.button_dropoff.config(bg="#30F38E")
            
            if self.terminated or self.truncated:
                self.episodes_reward.append(self.tot_reward)
                self.episodes_arr.append(self.episode_number)
                steps = self.taxi.get_steps()
                self.episode_steps.append(steps)
                self.episodes_type.append(self.episode_type_string)
                print(f"Episode tot_reward: {self.tot_reward}")

                self.episode_number+=1
                self.tot_reward = 0
                self.check_buttons_suggestions_clicked = False 

                self.label_comunication.config(text = "Episode terminated, please restart a match!")

    def button_right_clicked(self):

        if not self.terminated and not self.truncated:
            if self.check_buttons_suggestions_clicked:
                taxi_predict_2 = deepcopy(self.taxi_predict)
                self.reward_predict, steps_predict, terminated_predict, truncated_predict, observation_predict = taxi_predict_2.select_action(self.action_predict)

                self.tot_reward_predict = self.tot_reward + self.reward_predict

                self.taxi_predict.select_action(2)

            reward, steps, self.terminated, self.truncated, observation = self.taxi.select_action(2)
            image_load = self.taxi.render_image()
            self.image_render = ImageTk.PhotoImage(image_load, master=self)
            self.label_1.config(image=self.image_render)
            self.update_graph(reward, steps, 2, observation)

            if self.check_buttons_suggestions_clicked:
                self.button_up.config(bg="#F38630")
                self.button_down.config(bg="#F38630")
                self.button_right.config(bg="#F38630")
                self.button_left.config(bg="#F38630")
                self.button_pickup.config(bg="#F38630")
                self.button_dropoff.config(bg="#F38630")

                observation = int(self.taxi.get_state())
                self.action_predict = self.taxi.learnedAction(observation)

                if self.action_predict == 1:
                    self.button_up.config(bg="#30F38E")
                if self.action_predict == 0:
                    self.button_down.config(bg="#30F38E")
                if self.action_predict == 2:
                    self.button_right.config(bg="#30F38E")
                if self.action_predict == 3:
                    self.button_left.config(bg="#30F38E")
                if self.action_predict == 4:
                    self.button_pickup.config(bg="#30F38E")
                if self.action_predict == 5:
                    self.button_dropoff.config(bg="#30F38E")
            
            if self.terminated or self.truncated:
                self.episodes_reward.append(self.tot_reward)
                self.episodes_arr.append(self.episode_number)
                steps = self.taxi.get_steps()
                self.episode_steps.append(steps)
                self.episodes_type.append(self.episode_type_string)
                print(f"Episode tot_reward: {self.tot_reward}")

                self.episode_number+=1
                self.tot_reward = 0
                self.check_buttons_suggestions_clicked = False 

                self.label_comunication.config(text = "Episode terminated, please restart a match!")

    def button_left_clicked(self):

        if not self.terminated and not self.truncated:
            if self.check_buttons_suggestions_clicked:
                taxi_predict_2 = deepcopy(self.taxi_predict)
                self.reward_predict, steps_predict, terminated_predict, truncated_predict, observation_predict = taxi_predict_2.select_action(self.action_predict)

                self.tot_reward_predict = self.tot_reward + self.reward_predict

                self.taxi_predict.select_action(3)

            reward, steps, self.terminated, self.truncated, observation = self.taxi.select_action(3)
            image_load = self.taxi.render_image()
            self.image_render = ImageTk.PhotoImage(image_load, master=self)
            self.label_1.config(image=self.image_render)
            self.update_graph(reward, steps, 3, observation)

            if self.check_buttons_suggestions_clicked:
                self.button_up.config(bg="#F38630")
                self.button_down.config(bg="#F38630")
                self.button_right.config(bg="#F38630")
                self.button_left.config(bg="#F38630")
                self.button_pickup.config(bg="#F38630")
                self.button_dropoff.config(bg="#F38630")

                observation = int(self.taxi.get_state())
                self.action_predict = self.taxi.learnedAction(observation)

                if self.action_predict == 1:
                    self.button_up.config(bg="#30F38E")
                if self.action_predict == 0:
                    self.button_down.config(bg="#30F38E")
                if self.action_predict == 2:
                    self.button_right.config(bg="#30F38E")
                if self.action_predict == 3:
                    self.button_left.config(bg="#30F38E")
                if self.action_predict == 4:
                    self.button_pickup.config(bg="#30F38E")
                if self.action_predict == 5:
                    self.button_dropoff.config(bg="#30F38E")
            
            if self.terminated or self.truncated:
                self.episodes_reward.append(self.tot_reward)
                self.episodes_arr.append(self.episode_number)
                steps = self.taxi.get_steps()
                self.episode_steps.append(steps)
                self.episodes_type.append(self.episode_type_string)
                print(f"Episode tot_reward: {self.tot_reward}")

                self.episode_number+=1
                self.tot_reward = 0
                self.check_buttons_suggestions_clicked = False 

                self.label_comunication.config(text = "Episode terminated, please restart a match!")
    
    def button_pickup_clicked(self):

        if not self.terminated and not self.truncated:
            if self.check_buttons_suggestions_clicked:
                taxi_predict_2 = deepcopy(self.taxi_predict)
                self.reward_predict, steps_predict, terminated_predict, truncated_predict, observation_predict = taxi_predict_2.select_action(self.action_predict)

                self.tot_reward_predict = self.tot_reward + self.reward_predict

                self.taxi_predict.select_action(4)

            reward, steps, self.terminated, self.truncated, observation = self.taxi.select_action(4)
            image_load = self.taxi.render_image()
            self.image_render = ImageTk.PhotoImage(image_load, master=self)
            self.label_1.config(image=self.image_render)
            self.update_graph(reward, steps, 4, observation)

            if self.check_buttons_suggestions_clicked:
                self.button_up.config(bg="#F38630")
                self.button_down.config(bg="#F38630")
                self.button_right.config(bg="#F38630")
                self.button_left.config(bg="#F38630")
                self.button_pickup.config(bg="#F38630")
                self.button_dropoff.config(bg="#F38630")

                observation = int(self.taxi.get_state())
                self.action_predict = self.taxi.learnedAction(observation)

                if self.action_predict == 1:
                    self.button_up.config(bg="#30F38E")
                if self.action_predict == 0:
                    self.button_down.config(bg="#30F38E")
                if self.action_predict == 2:
                    self.button_right.config(bg="#30F38E")
                if self.action_predict == 3:
                    self.button_left.config(bg="#30F38E")
                if self.action_predict == 4:
                    self.button_pickup.config(bg="#30F38E")
                if self.action_predict == 5:
                    self.button_dropoff.config(bg="#30F38E")
            
            if self.terminated or self.truncated:
                self.episodes_reward.append(self.tot_reward)
                self.episodes_arr.append(self.episode_number)
                steps = self.taxi.get_steps()
                self.episode_steps.append(steps)
                self.episodes_type.append(self.episode_type_string)
                print(f"Episode tot_reward: {self.tot_reward}")

                self.episode_number+=1
                self.tot_reward = 0
                self.check_buttons_suggestions_clicked = False 

                self.label_comunication.config(text = "Episode terminated, please restart a match!")

    def button_dropoff_clicked(self):

        if not self.terminated and not self.truncated:
            if self.check_buttons_suggestions_clicked:
                taxi_predict_2 = deepcopy(self.taxi_predict)
                self.reward_predict, steps_predict, terminated_predict, truncated_predict, observation_predict = taxi_predict_2.select_action(self.action_predict)

                self.tot_reward_predict = self.tot_reward + self.reward_predict

                self.taxi_predict.select_action(5)

            reward, steps, self.terminated, self.truncated, observation = self.taxi.select_action(5)
            image_load = self.taxi.render_image()
            self.image_render = ImageTk.PhotoImage(image_load, master=self)
            self.label_1.config(image=self.image_render)
            self.update_graph(reward, steps, 5, observation)

            if self.check_buttons_suggestions_clicked:
                self.button_up.config(bg="#F38630")
                self.button_down.config(bg="#F38630")
                self.button_right.config(bg="#F38630")
                self.button_left.config(bg="#F38630")
                self.button_pickup.config(bg="#F38630")
                self.button_dropoff.config(bg="#F38630")

                observation = int(self.taxi.get_state())
                self.action_predict = self.taxi.learnedAction(observation)

                if self.action_predict == 1:
                    self.button_up.config(bg="#30F38E")
                if self.action_predict == 0:
                    self.button_down.config(bg="#30F38E")
                if self.action_predict == 2:
                    self.button_right.config(bg="#30F38E")
                if self.action_predict == 3:
                    self.button_left.config(bg="#30F38E")
                if self.action_predict == 4:
                    self.button_pickup.config(bg="#30F38E")
                if self.action_predict == 5:
                    self.button_dropoff.config(bg="#30F38E")
            
            if self.terminated or self.truncated:
                self.episodes_reward.append(self.tot_reward)
                self.episodes_arr.append(self.episode_number)
                steps = self.taxi.get_steps()
                self.episode_steps.append(steps)
                self.episodes_type.append(self.episode_type_string)
                print(f"Episode tot_reward: {self.tot_reward}")

                self.episode_number+=1
                self.tot_reward = 0

                self.check_buttons_suggestions_clicked = False 

                self.label_comunication.config(text = "Episode terminated, please restart a match!")   

    def button_home_clicked(self):
        
        if self.environment_started:
            #replace preliminary buttons
            self.image_render = None
            #self.label_1.place_forget()
            self.button_save.place_forget()
            self.button_new_episode.place_forget()
            self.button_qlearning.place_forget()
            self.button_score.place_forget()
            self.button_episode_suggestions.place_forget()
            self.button_start.place(
                x=180.0,
                y=534.0,
                width=240.0,
                height=70.0
            )

            #reset button color
            self.button_up.config(bg="#F38630")
            self.button_down.config(bg="#F38630")
            self.button_right.config(bg="#F38630")
            self.button_left.config(bg="#F38630")
            self.button_pickup.config(bg="#F38630")
            self.button_dropoff.config(bg="#F38630")

            self.label_comunication.config(text = "")
            self.label_comunication.place(x=43, y=590)

            self.tot_reward = 0
            self.check_field = False #check field setting, enable arrows 
            self.episode_number = 0
            self.episodes_arr = [] #store the number of episode
            self.episodes_reward = [] #save the tot_reward of every episode
            self.episode_steps = [] #save the number of steps of each episode
            self.episodes_type = []
            self.terminated = False
            self.truncated = False
            self.check_buttons_suggestions_clicked = False

            self.environment_started = False

            #forget buttons
            self.button_up.place(
                x=250.0,
                y=624.0,
                width=100.0,
                height=100.0
            )

            self.button_right.place(
                x=382.0,
                y=756.0,
                width=100.0,
                height=100.0
            )

            self.button_down.place(
                x=250.0,
                y=756.0,
                width=100.0,
                height=100.0
            )

            self.button_left.place(
                x=118.0,
                y=756.0,
                width=100.0,
                height=100.0
            )

            self.button_pickup.place(
                x=514.0,
                y=624.0,
                width=150.0,
                height=100.0
            )

            self.button_dropoff.place(
                x=514.0,
                y=756.0,
                width=150.0,
                height=100.0
            )

            self.button_up.place_forget()
            self.button_down.place_forget()
            self.button_left.place_forget()
            self.button_right.place_forget()
            self.button_pickup.place_forget()
            self.button_dropoff.place_forget()

            #reset predict graph and variables
            self.tot_reward_predict = 0
        
        #reset graph
        self.x_data_tot_reward = []  
        self.y_data_tot_reward = []
        self.line_tot_reward.set_ydata(self.y_data_tot_reward)
        self.line_tot_reward.set_xdata(self.x_data_tot_reward)  
        
        self.x_data_rew = []  
        self.y_data_rew = []
        self.line_rew.set_ydata(self.y_data_rew)
        self.line_rew.set_xdata(self.x_data_rew)  
        
        self.x_data_steer = []  
        self.y_data_steer = []
        self.line_steer.set_ydata(self.y_data_steer)
        self.line_steer.set_xdata(self.x_data_steer)  

        self.x_data_tot_reward_predict = []  
        self.y_data_tot_reward_predict = []  
        self.line_tot_reward_predict.set_ydata(self.y_data_tot_reward_predict)
        self.line_tot_reward_predict.set_xdata(self.x_data_tot_reward_predict)

        self.x_data_rew_predict = []  
        self.y_data_rew_predict = []  
        self.line_rew_predict.set_ydata(self.y_data_rew_predict)
        self.line_rew_predict.set_xdata(self.x_data_rew_predict)

        self.x_data_steer_predict = []  
        self.y_data_steer_predict = []
        self.line_steer_predict.set_ydata(self.y_data_steer_predict)
        self.line_steer_predict.set_xdata(self.x_data_steer_predict)

        plt.draw() 
        self.canvas.draw() 
        self.update()
            
        self.controller.show_frame("Home")

            
