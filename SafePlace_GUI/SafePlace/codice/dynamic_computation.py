# -*- coding: utf-8 -*-

"""
Online Model Adaptation in Monte Carlo Tree Search Planning

This file is part of free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

It is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with the code.  If not, see <http://www.gnu.org/licenses/>.
"""

import time
import utils
import stats
from safeplace_env import SafePlaceEnv
from action_data import ActionData
from state_data import StateData
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
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, messagebox
from pathlib import Path
import csv
from tkinter import font
from datetime import datetime

import sys
from os.path import abspath, dirname
# add PPO module to sys.path
from PPO.ppo_predict import PPO_predict

#global variables
step = 0
state = None
tot_reward = 0
next_state = None
action_predict = None

check_suggestions_enable = False #True if suggestion mode
check_ppo_policy_clicked = False #used for disable the buttons
check_stop_clicked = False #if TRUE button stop is clicked
check_ppo_policy_first_time = True
check_episode_terminated = False #set True if an episode is terminated
check_ppo_score_clicked = False #ppo_score button clicked
score_string = None

current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") #used for save the report


def on_button_human_click():
    '''button human mode, containds the instructions for human mode match
    contains also the instructions for suggestions, the suggestions are enabled by the suggestions button
    if the player clicks suggestions button, it enables the check_suggestions_enable variable and calls on_button human click
    then in on_button_human_click the suggestion are managed'''
    global button_home
    global action_predict
    global check_ppo_score_clicked

    #show home button and PPO score
    button_home.place(
        x=1450,
        y=155,
        width=353.9110412597656,
        height=44.88607406616211
    )
    button_ppo_score.place(
        x=1050,
        y=155,
        width=353.9110412597656,
        height=44.88607406616211
    )

    #show the initial state
    global label
    label.config(text="Time: "+str(state.hour)+":"+str(state.minute)+", People: "+str(state.people)+ ", Total reward: "+str(tot_reward))

    #hide mode buttons
    global button_human, button_human_suggestions, button_ppo_policy
    button_human.place_forget()
    button_human_suggestions.place_forget()
    button_ppo_policy.place_forget()

    #place actions buttons
    global button_1, button_2, button_3, button_4, button_5, button_6, button_7, button_8
    button_1.place(
        x=46.0889892578125,
        y=565.0,
        width=353.9110412597656,
        height=44.88607406616211
    )

    button_2.place(
        x=45.0,
        y=614.8734130859375,
        width=353.9110412597656,
        height=44.88607406616211
    )

    button_3.place(
        x=45.0,
        y=664.746826171875,
        width=353.9110412597656,
        height=44.88607406616211
    )

    button_4.place(
        x=45.0,
        y=714.6202392578125,
        width=353.9110412597656,
        height=44.88607406616211
    )

    button_5.place(
        x=45.0,
        y=764.49365234375,
        width=353.9110412597656,
        height=44.88607406616211
    )

    button_6.place(
        x=46.0889892578125,
        y=814.3670654296875,
        width=353.9110412597656,
        height=44.88607406616211
    )

    button_7.place(
        x=45.0,
        y=864.240478515625,
        width=353.9110412597656,
        height=44.88607406616211
    )

    button_8.place(
        x=45.0,
        y=914.1139526367188,
        width=353.9110412597656,
        height=44.88607406616211
    )

    print(f"Check_suggestions_enable: {check_suggestions_enable}")

    #predict action and set button predict colour
    if check_suggestions_enable:
        ppo_agent = PPO_predict()
        action_predict = ppo_agent.get_predict_action(state)
        print(f"            \nStep:{step}: Action predict = {action_predict}\n") 
        set_buttons_action_predict(str(action_predict)) #set prediction


def on_button_human_suggestions_click(): 
    '''The user click human with suggestions button
    the instructions are the same as Human but the check_suggestions_enable are set to True'''
    global check_suggestions_enable

    check_suggestions_enable = True #enable suggestions
    on_button_human_click() #the instructions are the same

def on_button_ppo_policy_click():
    '''The user click ppo policy, set the buttons color and run PPO policy'''
    global step, state, tot_reward, check_ppo_policy_clicked, check_stop_clicked, check_ppo_policy_first_time, button_stop
    global x_data_temp_in, y_data_temp_in, line_temp_in, ax1, x_data_rew, y_data_rew, line_rew, ax2, x_data_co2, y_data_co2, line_co2, ax3 
    global x_data_temp_out, y_data_temp_out, line_temp_out, x_data_voc, y_data_voc, line_voc, ax4
    global canvas, label_1, label_2, label_3, vent_off, wind_off, san_off, vent_high_on,vent_low_on, wind_on, san_on
    global x_data_air_quality_reward, y_data_air_quality_reward, line_air_quality_reward
    global x_data_comfort_reward, y_data_comfort_reward, line_comfort_reward, x_data_energy_reward, y_data_energy_reward, line_energy_reward, ax6
    
    
    if check_ppo_policy_first_time:
        
        check_ppo_policy_clicked = True #to disable the buttons
        
        #show start/stop button
        button_stop.place(
        x=1000,
        y=155,
        width=353.9110412597656,
        height=44.88607406616211
        )

        #show the initial state
        global label
        label.config(text="Time: "+str(state.hour)+":"+str(state.minute)+", People: "+str(state.people)+ ", Total reward: "+str(tot_reward))

        #show action buttons
        global button_1, button_2, button_3, button_4, button_5, button_6, button_7, button_8
        global button_image_1, button_image_2, button_image_3, button_image_4, button_image_5, button_image_6, button_image_7, button_image_8
        button_1.place(
            x=46.0889892578125,
            y=565.0,
            width=353.9110412597656,
            height=44.88607406616211
        )

        button_2.place(
            x=45.0,
            y=614.8734130859375,
            width=353.9110412597656,
            height=44.88607406616211
        )

        button_3.place(
            x=45.0,
            y=664.746826171875,
            width=353.9110412597656,
            height=44.88607406616211
        )

        button_4.place(
            x=45.0,
            y=714.6202392578125,
            width=353.9110412597656,
            height=44.88607406616211
        )

        button_5.place(
            x=45.0,
            y=764.49365234375,
            width=353.9110412597656,
            height=44.88607406616211
        )

        button_6.place(
            x=46.0889892578125,
            y=814.3670654296875,
            width=353.9110412597656,
            height=44.88607406616211
        )

        button_7.place(
            x=45.0,
            y=864.240478515625,
            width=353.9110412597656,
            height=44.88607406616211
        )

        button_8.place(
            x=45.0,
            y=914.1139526367188,
            width=353.9110412597656,
            height=44.88607406616211
        )

        #hide mode buttons
        global button_human, button_human_suggestions
        button_human.place_forget()
        button_human_suggestions.place_forget()
        button_ppo_policy.place_forget()

        check_ppo_policy_first_time = False

    #run PPO policy
    while not state.is_terminal:
        if check_stop_clicked: #button stop clicked
            return
        
        #reset color buttons
        button_image_1 = PhotoImage(
            file=relative_to_assets("button_1.png"), master=window)
        button_1.config(image=button_image_1)

        button_image_2 = PhotoImage(
            file=relative_to_assets("button_2.png"), master=window)
        button_2.config(image=button_image_2)

        button_image_3 = PhotoImage(
            file=relative_to_assets("button_3.png"), master=window)
        button_3.config(image=button_image_3)

        button_image_4 = PhotoImage(
            file=relative_to_assets("button_4.png"), master=window)
        button_4.config(image=button_image_4)

        button_image_5 = PhotoImage(
            file=relative_to_assets("button_5.png"), master=window)
        button_5.config(image=button_image_5)

        button_image_6 = PhotoImage(
            file=relative_to_assets("button_6.png"), master=window)
        button_6.config(image=button_image_6)

        button_image_7 = PhotoImage(
            file=relative_to_assets("button_7.png"), master=window)
        button_7.config(image=button_image_7)

        button_image_8 = PhotoImage(
            file=relative_to_assets("button_8.png"), master=window)
        button_8.config(image=button_image_8)

        #set PPO button color and predict action
        ppo_agent = PPO_predict()
        action_predict = ppo_agent.get_predict_action(state)
        print(f"            \nStep:{step}: Action predict = {action_predict}\n") 

        
        button_number = 0
        if str(action_predict) == "ActionData.ALL_OFF":
            button_image_1 = PhotoImage(
                file=relative_to_assets("button_1_green.png"), master=window)
            button_1.config(image=button_image_1)
            button_number = 1
        
        if str(action_predict) == "ActionData.WINDOW_ON":
            button_image_2 = PhotoImage(
            file=relative_to_assets("button_2_green.png"), master=window)
            button_2.config(image=button_image_2)
            button_number = 2
        
        if str(action_predict) == "ActionData.VENT_LOW_ON":
            button_image_3 = PhotoImage(
            file=relative_to_assets("button_3_green.png"), master=window)
            button_3.config(image=button_image_3)
            button_number = 3
        
        if str(action_predict) == "ActionData.VENT_HIGH_ON":
            button_image_4 = PhotoImage(
            file=relative_to_assets("button_4_green.png"), master=window)
            button_4.config(image=button_image_4)
            button_number = 4
        
        if str(action_predict) == "ActionData.SANITIZER_ON":
            button_image_5 = PhotoImage(
            file=relative_to_assets("button_5_green.png"), master=window)
            button_5.config(image=button_image_5)
            button_number = 5
        
        if str(action_predict) == "ActionData.WINDOW_SANITIZER_ON":
            button_image_6 = PhotoImage(
            file=relative_to_assets("button_6_green.png"), master=window)
            button_6.config(image=button_image_6)
            button_number = 6
    
        if str(action_predict) == "ActionData.VENT_LOW_SANITIZER_ON":
            button_image_7 = PhotoImage(
            file=relative_to_assets("button_7_green.png"), master=window)
            button_7.config(image=button_image_7)
            button_number = 7
        
        if str(action_predict) == "ActionData.VENT_HIGH_SANITIZER_ON":
            button_image_8 = PhotoImage(
            file=relative_to_assets("button_8_green.png"), master=window)
            button_8.config(image=button_image_8)
            button_number = 8

        select_action(action_predict, button_number)

        time.sleep(1.5)
    
    if state.is_terminal:

        button_home.place(
        x=1450,
        y=155,
        width=353.9110412597656,
        height=44.88607406616211
    )

def on_button_stop_click():
    #stop/play the PPO, it uses a check_stop_clicked variable 
    global check_stop_clicked, button_home
    if not check_stop_clicked:
        check_stop_clicked = True
        #show home button
        button_home.place(
            x=1450,
            y=155,
            width=353.9110412597656,
            height=44.88607406616211
        )
    else: 
        check_stop_clicked = False
        button_home.place_forget()
        on_button_ppo_policy_click()


#select an action based on buttons
def select_action(a, button_number):
    '''Select the action chosen, update the graphs, show suggestions if they are enable'''
    global step, state, tot_reward
    global x_data_temp_in, y_data_temp_in, line_temp_in, ax1, x_data_rew, y_data_rew, line_rew, ax2, x_data_co2, y_data_co2, line_co2, ax3 
    global x_data_temp_out, y_data_temp_out, line_temp_out, x_data_voc, y_data_voc, line_voc, ax4, x_data_air_quality_reward, y_data_air_quality_reward, line_air_quality_reward
    global x_data_comfort_reward, y_data_comfort_reward, line_comfort_reward, x_data_energy_reward, y_data_energy_reward, line_energy_reward, ax6
    global check_suggestions_enable, check_ppo_score_clicked, score_string, vertical_line_people, action_predict, x_data_ipo_temp_in, y_data_ipo_temp_in, line_ipo_temp_in
    global x_data_ipo_rew, y_data_ipo_rew, line_ipo_rew, x_data_ipo_co2, y_data_ipo_co2, line_ipo_co2, x_data_ipo_voc, y_data_ipo_voc, line_ipo_voc
    global x_data_ipo_air_quality_reward, y_data_ipo_air_quality_reward, line_ipo_air_quality_reward, x_data_ipo_comfort_reward, y_data_ipo_comfort_reward, line_ipo_comfort_reward, x_data_ipo_energy_reward, y_data_ipo_energy_reward, line_ipo_energy_reward
    global canvas, label_1, label_2, label_3, vent_off, wind_off, san_off, vent_high_on,vent_low_on, wind_on, san_on, label
    
    
    if check_suggestions_enable:
        global button_1, button_2, button_3, button_4, button_5, button_6, button_7, button_8
        global button_image_1, button_image_2, button_image_3, button_image_4, button_image_5, button_image_6, button_image_7, button_image_8
        global ax5, check_episode_terminated
        
        #reset button color
        button_image_1 = PhotoImage(
            file=relative_to_assets("button_1.png"), master=window)
        button_1.config(image=button_image_1)

        button_image_2 = PhotoImage(
            file=relative_to_assets("button_2.png"), master=window)
        button_2.config(image=button_image_2)

        button_image_3 = PhotoImage(
            file=relative_to_assets("button_3.png"), master=window)
        button_3.config(image=button_image_3)

        button_image_4 = PhotoImage(
            file=relative_to_assets("button_4.png"), master=window)
        button_4.config(image=button_image_4)

        button_image_5 = PhotoImage(
            file=relative_to_assets("button_5.png"), master=window)
        button_5.config(image=button_image_5)

        button_image_6 = PhotoImage(
            file=relative_to_assets("button_6.png"), master=window)
        button_6.config(image=button_image_6)

        button_image_7 = PhotoImage(
            file=relative_to_assets("button_7.png"), master=window)
        button_7.config(image=button_image_7)

        button_image_8 = PhotoImage(
            file=relative_to_assets("button_8.png"), master=window)
        button_8.config(image=button_image_8)

    
    next_state, rewards = utils.env.do_transition(state, a) # Next state
    
    prev_state = state #set prediction state
    prev_action_predict = action_predict #set predict action

    #save some data and show the action
    if utils.step_stats:
        stats.step_stats_record_data(state, a, rewards)

    if utils.verbose:
        print('Action: ' + str(a))
        print('Real state: ' + str(next_state))
        print('Reward: '+ str(rewards))

    #set variables
    temp_in = next_state.temp_in
    temp_out = next_state.temp_out

    state = next_state

    tot_reward += rewards[0]
    
    #update state label
    if check_ppo_score_clicked:
        label.config(text="Time: "+str(state.hour)+":"+str(state.minute)+", People: "+str(state.people)+ ", Total reward: "+str(round(tot_reward,2)) + score_string)
    else:
        label.config(text="Time: "+str(state.hour)+":"+str(state.minute)+", People: "+str(state.people)+ ", Total reward: "+str(round(tot_reward,2)))
    
    #if state.is_terminal write to csv and set terminal variable in order to disable the buttons
    if state.is_terminal:
        if check_suggestions_enable:
            type_mode = "suggestions"
            write_to_csv(type_mode)
        elif check_ppo_policy_clicked:
            type_mode = "ppo_policy"
            write_to_csv(type_mode)
        else:
            type_mode = "human"
            write_to_csv(type_mode)
        
        check_episode_terminated = True

        label.config(text = label.cget("text")+", TERMINATED")

    #update images
    if button_number == 1:
        label_1.config(image=vent_off)
        label_2.config(image=wind_off)
        label_3.config(image=san_off)
    if button_number == 2:
        label_1.config(image=vent_off)
        label_2.config(image=wind_on)
        label_3.config(image=san_off)
    if button_number == 3:
        label_1.config(image=vent_low_on)
        label_2.config(image=wind_off)
        label_3.config(image=san_off)
    if button_number == 4:
        label_1.config(image=vent_high_on)
        label_2.config(image=wind_off)
        label_3.config(image=san_off)
    if button_number == 5:
        label_1.config(image=vent_off)
        label_2.config(image=wind_off)
        label_3.config(image=san_on)
    if button_number == 6:
        label_1.config(image=vent_off)
        label_2.config(image=wind_on)
        label_3.config(image=san_on)
    if button_number == 7:
        label_1.config(image=vent_low_on)
        label_2.config(image=wind_off)
        label_3.config(image=san_on)
    if button_number == 8:
        label_1.config(image=vent_high_on)
        label_2.config(image=wind_off)
        label_3.config(image=san_on)
    

    #graph update
    x_data_temp_in.append(step)
    y_data_temp_in.append(temp_in)
    line_temp_in.set_ydata(y_data_temp_in)
    line_temp_in.set_xdata(x_data_temp_in)

    x_data_temp_out.append(step)
    y_data_temp_out.append(temp_out)
    line_temp_out.set_ydata(y_data_temp_out)
    line_temp_out.set_xdata(x_data_temp_out)
    ax1.relim()
    ax1.autoscale_view(True, True, True)
   
    x_data_rew.append(step)
    y_data_rew.append(rewards[0])
    line_rew.set_ydata(y_data_rew)
    line_rew.set_xdata(x_data_rew)
    ax2.relim()
    ax2.autoscale_view(True, True, True)
    
    x_data_co2.append(step)
    y_data_co2.append(state.co2)
    line_co2.set_ydata(y_data_co2)
    line_co2.set_xdata(x_data_co2)
    ax3.relim()
    ax3.autoscale_view(True, True, True)
    
    x_data_voc.append(step)  
    y_data_voc.append(state.voc) 
    line_voc.set_ydata(y_data_voc)
    line_voc.set_xdata(x_data_voc)
    ax4.relim()
    ax4.autoscale_view(True, True, True)

    x_data_air_quality_reward.append(step)
    y_data_air_quality_reward.append(rewards[1])
    line_air_quality_reward.set_ydata(y_data_air_quality_reward)
    line_air_quality_reward.set_xdata(x_data_air_quality_reward)

    x_data_comfort_reward.append(step) 
    y_data_comfort_reward.append(rewards[2])  
    line_comfort_reward.set_ydata(y_data_comfort_reward)
    line_comfort_reward.set_xdata(x_data_comfort_reward)
    
    x_data_energy_reward.append(step) 
    y_data_energy_reward.append(rewards[3])
    line_energy_reward.set_ydata(y_data_energy_reward)
    line_energy_reward.set_xdata(x_data_energy_reward)
    ax6.relim()
    ax6.autoscale_view(True, True, True)  
    
    vertical_line_people.remove()
    vertical_line_people = ax5.axvline(x=step, color='orange', linestyle='--')

    # move xlim visualization 
    if step <= 9:
        ax1.set_xlim(-0.5, 10)
        ax2.set_xlim(-0.5, 10)
        ax3.set_xlim(-0.5, 10)
        ax4.set_xlim(-0.5, 10)
        ax6.set_xlim(-0.5, 10)
    else:
        ax1.set_xlim(step -10, step+1)
        ax2.set_xlim(step -10, step+1)
        ax3.set_xlim(step -10, step+1)
        ax4.set_xlim(step -10, step+1)
        ax6.set_xlim(step -10, step+1)
        x_data_temp_in = x_data_temp_in[-10:]
        y_data_temp_in = y_data_temp_in[-10:]
        x_data_temp_out = x_data_temp_out[-10:]
        y_data_temp_out = y_data_temp_out[-10:]
        x_data_rew = x_data_rew[-10:]
        y_data_rew = y_data_rew[-10:]
        x_data_co2 = x_data_co2[-10:]
        y_data_co2 = y_data_co2[-10:]
        x_data_voc = x_data_voc[-10:]
        y_data_voc = y_data_voc[-10:]
        x_data_air_quality_reward = x_data_air_quality_reward[-10:]
        y_data_air_quality_reward = y_data_air_quality_reward[-10:]
        x_data_comfort_reward = x_data_comfort_reward[-10:]
        y_data_comfort_reward = y_data_comfort_reward[-10:]
        x_data_energy_reward = x_data_energy_reward[-10:]
        y_data_energy_reward = y_data_energy_reward[-10:]

    plt.draw()
    window.update()

    print(f"Check_suggestions_enable: {check_suggestions_enable}")
    
    #if suggestions enable set the points on the graphs of the action chosen by the PPO(orange points)
    if check_suggestions_enable:
        ipotetical_state, ipotetical_rewards = utils.env.do_transition(prev_state, prev_action_predict)

        #set point ppo predict
        x_data_ipo_temp_in.append(step)
        y_data_ipo_temp_in.append(ipotetical_state.temp_in)
        x_data_ipo_temp_in = x_data_ipo_temp_in[-1:]
        y_data_ipo_temp_in = y_data_ipo_temp_in[-1:]
        line_ipo_temp_in.set_xdata(x_data_ipo_temp_in)
        line_ipo_temp_in.set_ydata(y_data_ipo_temp_in)
        ax1.relim()
        ax1.autoscale_view(True,True,True)

        x_data_ipo_rew.append(step)
        y_data_ipo_rew.append(ipotetical_rewards[0])
        x_data_ipo_rew = x_data_ipo_rew[-1:]
        y_data_ipo_rew =y_data_ipo_rew[-1:]
        line_ipo_rew.set_xdata(x_data_ipo_rew)
        line_ipo_rew.set_ydata(y_data_ipo_rew)
        ax2.relim()
        ax2.autoscale_view(True,True,True)

        x_data_ipo_co2.append(step)
        y_data_ipo_co2.append(ipotetical_state.co2)
        x_data_ipo_co2 = x_data_ipo_co2[-1:]
        y_data_ipo_co2 =y_data_ipo_co2[-1:]
        line_ipo_co2.set_xdata(x_data_ipo_co2)
        line_ipo_co2.set_ydata(y_data_ipo_co2)
        ax3.relim()
        ax3.autoscale_view(True,True,True)

        x_data_ipo_voc.append(step)
        y_data_ipo_voc.append(ipotetical_state.voc)
        x_data_ipo_voc = x_data_ipo_voc[-1:]
        y_data_ipo_voc =y_data_ipo_voc[-1:]
        line_ipo_voc.set_xdata(x_data_ipo_voc)
        line_ipo_voc.set_ydata(y_data_ipo_voc)
        ax4.relim()
        ax4.autoscale_view(True,True,True)

        x_data_ipo_air_quality_reward.append(step)
        y_data_ipo_air_quality_reward.append(ipotetical_rewards[1])
        x_data_ipo_air_quality_reward = x_data_ipo_air_quality_reward[-1:]
        y_data_ipo_air_quality_reward =y_data_ipo_air_quality_reward[-1:]
        line_ipo_air_quality_reward.set_xdata(x_data_ipo_air_quality_reward)
        line_ipo_air_quality_reward.set_ydata(y_data_ipo_air_quality_reward)
        ax6.relim()
        ax6.autoscale_view(True,True,True)

        x_data_ipo_comfort_reward.append(step)
        y_data_ipo_comfort_reward.append(ipotetical_rewards[2])
        x_data_ipo_comfort_reward = x_data_ipo_comfort_reward[-1:]
        y_data_ipo_comfort_reward =y_data_ipo_comfort_reward[-1:]
        line_ipo_comfort_reward.set_xdata(x_data_ipo_comfort_reward)
        line_ipo_comfort_reward.set_ydata(y_data_ipo_comfort_reward)
        ax6.relim()
        ax6.autoscale_view(True,True,True)

        x_data_ipo_energy_reward.append(step)
        y_data_ipo_energy_reward.append(ipotetical_rewards[3])
        x_data_ipo_energy_reward = x_data_ipo_energy_reward[-1:]
        y_data_ipo_energy_reward =y_data_ipo_energy_reward[-1:]
        line_ipo_energy_reward.set_xdata(x_data_ipo_energy_reward)
        line_ipo_energy_reward.set_ydata(y_data_ipo_energy_reward)
        ax6.relim()
        ax6.autoscale_view(True,True,True)

        plt.draw()

        #predict next action to take
        ppo_agent = PPO_predict()
        action_predict = ppo_agent.get_predict_action(next_state)
        print(f"            \nStep:{step}: Action predict = {action_predict}\n")  
        set_buttons_action_predict(str(action_predict))

    #next step
    step +=1

    return state, a, rewards


def set_buttons_action_predict(action_predict):
    #set green button based to action predict by the ppo policy
    global button_1, button_2, button_3, button_4, button_5, button_6, button_7, button_8
    global button_image_1, button_image_2, button_image_3, button_image_4, button_image_5, button_image_6, button_image_7, button_image_8
    
    
    if action_predict == "ActionData.ALL_OFF":
        button_image_1 = PhotoImage(
            file=relative_to_assets("button_1_green.png"), master=window)
        button_1.config(image=button_image_1)
    
    if action_predict == "ActionData.WINDOW_ON":
        button_image_2 = PhotoImage(
            file=relative_to_assets("button_2_green.png"), master=window)
        button_2.config(image=button_image_2)
    
    if action_predict == "ActionData.VENT_LOW_ON":
        button_image_3 = PhotoImage(
            file=relative_to_assets("button_3_green.png"), master=window)
        button_3.config(image=button_image_3)
    
    if action_predict == "ActionData.VENT_HIGH_ON":
        button_image_4 = PhotoImage(
            file=relative_to_assets("button_4_green.png"), master=window)
        button_4.config(image=button_image_4)
    
    if action_predict == "ActionData.SANITIZER_ON":
        button_image_5 = PhotoImage(
            file=relative_to_assets("button_5_green.png"), master=window)
        button_5.config(image=button_image_5)
    
    if action_predict == "ActionData.WINDOW_SANITIZER_ON":
        button_image_6 = PhotoImage(
            file=relative_to_assets("button_6_green.png"), master=window)
        button_6.config(image=button_image_6)
   
    if action_predict == "ActionData.VENT_LOW_SANITIZER_ON":
        button_image_7 = PhotoImage(
            file=relative_to_assets("button_7_green.png"), master=window)
        button_7.config(image=button_image_7)
    
    if action_predict == "ActionData.VENT_HIGH_SANITIZER_ON":
        button_image_8 = PhotoImage(
            file=relative_to_assets("button_8_green.png"), master=window)
        button_8.config(image=button_image_8)
        

# Function on button event
# if in the functions is used to disable the buttons if the ppo is active or the episode is over
def on_button1_click():
    global check_ppo_policy_clicked, check_episode_terminated
    if check_ppo_policy_clicked or check_episode_terminated:
        return
    a = ActionData.ALL_OFF
    select_action(a, 1)

def on_button2_click():
    global check_ppo_policy_clicked, check_episode_terminated
    if check_ppo_policy_clicked or check_episode_terminated:
        return
    a = ActionData.WINDOW_ON
    select_action(a, 2)

def on_button3_click():
    global check_ppo_policy_clicked, check_episode_terminated
    if check_ppo_policy_clicked or check_episode_terminated:
        return
    a = ActionData.VENT_LOW_ON
    select_action(a, 3)

def on_button4_click():
    global check_ppo_policy_clicked, check_episode_terminated
    if check_ppo_policy_clicked or check_episode_terminated:
        return
    a = ActionData.VENT_HIGH_ON
    select_action(a, 4)

def on_button5_click():
    global check_ppo_policy_clicked, check_episode_terminated
    if check_ppo_policy_clicked or check_episode_terminated:
        return
    a = ActionData.SANITIZER_ON
    select_action(a, 5)

def on_button6_click():
    global check_ppo_policy_clicked, check_episode_terminated
    if check_ppo_policy_clicked or check_episode_terminated:
        return
    a = ActionData.WINDOW_SANITIZER_ON
    select_action(a, 6)

def on_button7_click():
    global check_ppo_policy_clicked, check_episode_terminated
    if check_ppo_policy_clicked or check_episode_terminated:
        return
    a = ActionData.VENT_LOW_SANITIZER_ON
    select_action(a, 7)

def on_button8_click():
    global check_ppo_policy_clicked, check_episode_terminated
    if check_ppo_policy_clicked or check_episode_terminated:
        return
    a = ActionData.VENT_HIGH_SANITIZER_ON
    select_action(a, 8)

def on_button_ppo_score_click():
    #show the score of one episode
    global pop_score 
    global check_ppo_score_clicked
    global score_string
    global label
    global button_ppo_score

    check_ppo_score_clicked = True

    #window
    pop_score = Toplevel(window)
    pop_score.title("PPO score")
    pop_score.geometry("450x120")
    pop_score.config(bg="white")

    label_pop_score = Label(pop_score, text = "PPO algorithm is running...", font = ("Inter", 20, "bold"), bg = "white")
    label_pop_score.grid(row=0, column=0, padx=10, pady=20)

    #set state for simulation
    state_sim = _initial_state
    total_reward_sim = 0

    time.sleep(1.5)
    window.update()

    #run the algorithm
    while not state_sim.is_terminal:
        ppo_agent_sim = PPO_predict()
        action_predict_sim = ppo_agent_sim.get_predict_action(state_sim)
        next_state_sim, reward_sim = utils.env.do_transition(state_sim, action_predict_sim)

        state_sim = next_state_sim
        total_reward_sim += reward_sim[0]

    #show the score
    label_pop_score.config(text = f"PPO score: {round(total_reward_sim,2)}")

    score_string = f", PPO score: {round(total_reward_sim,2)}"

    label.config(text=label.cget("text") + score_string)

    button_ppo_score.place_forget()
    

def on_button_info_click():
    #info button, show the info image
    global pop 
    pop = Toplevel(window)
    pop.title("Info popup")
    pop.geometry("850x850")
    pop.config(bg="white")

    global info1, label_pop1
    info1 = PhotoImage(file=relative_to_assets("info1.png"), master=pop)
    label_pop1 = Label(pop, image=info1)
    label_pop1.grid(row=0, column=0, padx=2)
    label_pop1.config(borderwidth=0)

    global button_image_succ, button_succ
    button_image_succ = PhotoImage(
        file=relative_to_assets("button_next.png"), master=pop)
    button_succ = Button(
        image=button_image_succ,
        borderwidth=0,
        highlightthickness=0,
        command=on_button_human_succ,
        relief="flat",
        master = pop
    )
    button_succ.grid(
        row=1,
        column=0,
    )

#click next in info window
def on_button_human_succ():
    global button_succ
    button_succ.grid_forget()
    global info1, label_pop1
    info1 = PhotoImage(file=relative_to_assets("info2.png"), master=pop)
    label_pop1.config(image = info1)


    
def on_button_home_click():
    #reset the interface
    global button_1, button_2, button_3, button_4, button_5, button_6, button_7, button_8, button_human, button_human_suggestions, button_ppo_policy, button_home
    global button_image_1, button_image_2, button_image_3, button_image_4, button_image_5, button_image_6, button_image_7, button_image_8
    global state, step, tot_reward
    global x_data_temp_in, y_data_temp_in, line_temp_in, ax1, x_data_rew, y_data_rew, line_rew, ax2, x_data_co2, y_data_co2, line_co2, ax3 
    global x_data_temp_out, y_data_temp_out, line_temp_out, x_data_voc, y_data_voc, line_voc, ax4, canvas, label_1, label_2, label_3, vent_off, wind_off, san_off, vent_high_on,vent_low_on, wind_on, san_on
    global x_data_air_quality_reward, y_data_air_quality_reward, line_air_quality_reward, x_data_comfort_reward, y_data_comfort_reward, line_comfort_reward
    global x_data_energy_reward, y_data_energy_reward, line_energy_reward, ax6, check_suggestions_enable, check_episode_terminated
    global vertical_line_people, check_ppo_policy_first_time, check_ppo_policy_clicked, check_stop_clicked, check_ppo_score_clicked
    global x_data_ipo_temp_in, y_data_ipo_temp_in, line_ipo_temp_in, x_data_ipo_rew, y_data_ipo_rew, line_ipo_rew, x_data_ipo_co2, y_data_ipo_co2, line_ipo_co2
    global x_data_ipo_voc, y_data_ipo_voc, line_ipo_voc, x_data_ipo_air_quality_reward, y_data_ipo_air_quality_reward, line_ipo_air_quality_reward
    global x_data_ipo_comfort_reward, y_data_ipo_comfort_reward, line_ipo_comfort_reward, x_data_ipo_energy_reward, y_data_ipo_energy_reward, line_ipo_energy_reward

    if check_ppo_policy_clicked:
        button_stop.place_forget()

    #reset variables
    check_suggestions_enable = False
    check_ppo_policy_first_time = True
    check_ppo_policy_clicked = False
    check_stop_clicked = False
    check_episode_terminated = False
    check_ppo_score_clicked = False

    #reset buttons colors
    button_image_1 = PhotoImage(
        file=relative_to_assets("button_1.png"), master=window)
    button_1.config(image=button_image_1)

    button_image_2 = PhotoImage(
        file=relative_to_assets("button_2.png"), master=window)
    button_2.config(image=button_image_2)

    button_image_3 = PhotoImage(
        file=relative_to_assets("button_3.png"), master=window)
    button_3.config(image=button_image_3)

    button_image_4 = PhotoImage(
        file=relative_to_assets("button_4.png"), master=window)
    button_4.config(image=button_image_4)

    button_image_5 = PhotoImage(
        file=relative_to_assets("button_5.png"), master=window)
    button_5.config(image=button_image_5)

    button_image_6 = PhotoImage(
        file=relative_to_assets("button_6.png"), master=window)
    button_6.config(image=button_image_6)

    button_image_7 = PhotoImage(
        file=relative_to_assets("button_7.png"), master=window)
    button_7.config(image=button_image_7)

    button_image_8 = PhotoImage(
        file=relative_to_assets("button_8.png"), master=window)
    button_8.config(image=button_image_8)
    
    button_1.place_forget()
    button_2.place_forget()
    button_3.place_forget()
    button_4.place_forget()
    button_5.place_forget()
    button_6.place_forget()
    button_7.place_forget()
    button_8.place_forget()

    #replace mode choice buttons
    button_human.place(
        x=335,
        y=155,
        width=353.9110412597656,
        height=44.88607406616211
    )
    button_human_suggestions.place(
        x=700,
        y=155,
        width=353.9110412597656,
        height=44.88607406616211
    )
    button_ppo_policy.place(
        x=1065,
        y=155,
        width=353.9110412597656,
        height=44.88607406616211
    )

    #initial state
    _initial_state = utils.initial_state()

    if utils.step_stats:
        s_stats_time = stats.step_stats_start()

    state = _initial_state

    #reset variables
    step = 0
    tot_reward = 0

    #reset graph
    x_data_temp_in = []  
    y_data_temp_in = [] 

    x_data_temp_out = []  
    y_data_temp_out = [] 

    x_data_rew = []  
    y_data_rew = [] 

    x_data_co2 = []  
    y_data_co2 = [] 

    x_data_voc = []  
    y_data_voc = [] 

    x_data_air_quality_reward = [] 
    y_data_air_quality_reward = []

    x_data_comfort_reward = [] 
    y_data_comfort_reward = []

    x_data_energy_reward = [] 
    y_data_energy_reward = []

    x_data_ipo_temp_in = []
    y_data_ipo_temp_in =[]
    line_ipo_temp_in.set_xdata(x_data_ipo_temp_in)
    line_ipo_temp_in.set_ydata(y_data_ipo_temp_in)

    x_data_ipo_rew = []
    y_data_ipo_rew =[]
    line_ipo_rew.set_xdata(x_data_ipo_rew)
    line_ipo_rew.set_ydata(y_data_ipo_rew)

    x_data_ipo_co2 = []
    y_data_ipo_co2 =[]
    line_ipo_co2.set_xdata(x_data_ipo_co2)
    line_ipo_co2.set_ydata(y_data_ipo_co2)

    x_data_ipo_voc = []
    y_data_ipo_voc =[]
    line_ipo_voc.set_xdata(x_data_ipo_voc)
    line_ipo_voc.set_ydata(y_data_ipo_voc)

    x_data_ipo_air_quality_reward = []
    y_data_ipo_air_quality_reward =[]
    line_ipo_air_quality_reward.set_xdata(x_data_ipo_air_quality_reward)
    line_ipo_air_quality_reward.set_ydata(y_data_ipo_air_quality_reward)

    x_data_ipo_comfort_reward = []
    y_data_ipo_comfort_reward =[]
    line_ipo_comfort_reward.set_xdata(x_data_ipo_comfort_reward)
    line_ipo_comfort_reward.set_ydata(y_data_ipo_comfort_reward)

    x_data_ipo_energy_reward = []
    y_data_ipo_energy_reward =[]
    line_ipo_energy_reward.set_xdata(x_data_ipo_energy_reward)
    line_ipo_energy_reward.set_ydata(y_data_ipo_energy_reward)

    #initial graph update
    x_data_temp_in.append(step)
    y_data_temp_in.append(state.temp_in)
    line_temp_in.set_ydata(y_data_temp_in)
    line_temp_in.set_xdata(x_data_temp_in)

    x_data_temp_out.append(step)
    y_data_temp_out.append(state.temp_out)
    line_temp_out.set_ydata(y_data_temp_out)
    line_temp_out.set_xdata(x_data_temp_out)
    ax1.relim()
    ax1.autoscale_view(True, True, True)
    
    x_data_co2.append(step)
    y_data_co2.append(state.co2)
    line_co2.set_ydata(y_data_co2)
    line_co2.set_xdata(x_data_co2)
    ax3.relim()
    ax3.autoscale_view(True, True, True)
    
    x_data_voc.append(step)
    y_data_voc.append(state.voc)
    line_voc.set_ydata(y_data_voc)
    line_voc.set_xdata(x_data_voc)
    ax4.relim()
    ax4.autoscale_view(True, True, True)

    line_rew.set_ydata(y_data_rew)
    line_rew.set_xdata(x_data_rew)
    ax2.relim()
    ax2.autoscale_view(True, True, True)

    line_air_quality_reward.set_ydata(y_data_air_quality_reward)
    line_air_quality_reward.set_xdata(x_data_air_quality_reward)
    ax6.relim()
    ax6.autoscale_view(True, True, True)

    line_comfort_reward.set_ydata(y_data_comfort_reward)
    line_comfort_reward.set_xdata(x_data_comfort_reward)
    ax6.relim()
    ax6.autoscale_view(True, True, True)

    line_energy_reward.set_ydata(y_data_energy_reward)
    line_energy_reward.set_xdata(x_data_energy_reward)
    ax6.relim()
    ax6.autoscale_view(True, True, True)

    vertical_line_people.remove()
    vertical_line_people = ax5.axvline(x=0, color='orange', linestyle='--')

    if step <= 9:
        ax1.set_xlim(-0.5, 10)
        ax2.set_xlim(-0.5, 10)
        ax3.set_xlim(-0.5, 10)
        ax4.set_xlim(-0.5, 10)
        ax6.set_xlim(-0.5, 10)
    else:
        ax1.set_xlim(step -10, step+1)
        ax2.set_xlim(step -10, step+1)
        ax3.set_xlim(step -10, step+1)
        ax4.set_xlim(step -10, step+1)
        ax6.set_xlim(step -10, step+1)

    plt.draw()
    
    #next step
    step +=1

    window.update()
    
    #replace images
    label_1.config(image=vent_off)
    label_2.config(image=wind_off)
    label_3.config(image=san_off)

    button_home.place_forget()
    button_ppo_score.place_forget()
    
    label.config(text = "Select a mode:")

def write_to_csv(type):
    global current_datetime, state, step, tot_reward
    
    #open csv and write on it
    try:
        nome_file = f'SafePlace/codice/report/SafePlace_report_{current_datetime}.csv'
        with open(nome_file, 'a', newline='') as file_csv:
        # Crea un oggetto scrittore CSV
            writer = csv.writer(file_csv)
            #wrote to csv
            writer.writerow([step,tot_reward,type])
        
    except IOError as e:
        print(f"Error CSV writing or reading")
    except Exception as e:
        print(f"Unknown error")


def close_window():
    window.destroy()
    exit()


##################################################################### MAIN #############################

if __name__ == '__main__':
    
    #################load the env
    # Load the reservation file relative to a single day
    utils.update_reservations(reservations_filepath=utils.reservation_profile_path)

    # Environment that uses the oracle as transition model
    env = SafePlaceEnv()

    # Framework initialization
    utils.initialize_all(env)

    _initial_state = utils.initial_state()

    if utils.step_stats:
        s_stats_time = stats.step_stats_start()

    state = _initial_state
    
    try:
        #set the csv file
        nome_file = f'SafePlace/codice/report/SafePlace_report_{current_datetime}.csv'
        
        with open(nome_file, mode='w', newline='') as file_csv:
            print("apertura")
        # Crea un oggetto scrittore CSV
            writer = csv.writer(file_csv)
    
            writer.writerow(['steps','total reward', 'type'])
    
    except IOError as e:
        print(f"Error CSV writing or reading")
    except Exception as e:
        print(f"Unknown error")
    
    #####################graph implementation with matplotlib
    global x_data_temp_in, y_data_temp_in, line_temp_in, x_data_temp_out, y_data_temp_out, line_temp_out, ax1
    global x_data_rew, y_data_rew, line_rew, ax2 
    global x_data_co2, y_data_co2, line_co2, ax3 
    global x_data_voc, y_data_voc, line_voc, ax4 
    global x_data_air_quality_reward, y_data_air_quality_reward, line_air_quality_reward
    global x_data_comfort_reward, y_data_comfort_reward, line_comfort_reward
    global x_data_energy_reward, y_data_energy_reward, line_energy_reward, ax6, vertical_line_people

    global x_data_ipo_temp_in, y_data_ipo_temp_in, line_ipo_temp_in, x_data_ipo_rew, y_data_ipo_rew, line_ipo_rew
    global x_data_ipo_co2, y_data_ipo_co2, line_ipo_co2, x_data_ipo_voc, y_data_ipo_voc, line_ipo_voc
    global x_data_ipo_air_quality_reward, y_data_ipo_air_quality_reward, line_ipo_air_quality_reward
    global x_data_ipo_comfort_reward, y_data_ipo_comfort_reward, line_ipo_comfort_reward
    global x_data_ipo_energy_reward, y_data_ipo_energy_reward, line_ipo_energy_reward
    
    fig, axs = plt.subplots(2, 3, figsize=(12,6))
    plt.subplots_adjust(hspace=0.5, left=0.1, right=0.95, wspace=0.3)
    
    ax1 = axs[0, 0]
    ax2 = axs[1, 0]
    ax3 = axs[0, 1]
    ax4 = axs[0, 2]
    ax5 = axs[1, 2]
    ax6 = axs[1, 1]

    #read csv file(reservations_test.csv) for people graph
    array_steps_csv = []
    array_people_csv = []

    with open('SafePlace/codice/datasets/reservations_test.csv', newline='') as csvfile:
        # csv reader object
        reader = csv.reader(csvfile)
        
        # Itera sulle righe del file CSV
        for i, row in enumerate(reader):
        # Salta la prima riga
            if i == 0:
                continue
            
            # steps column
            array_steps_csv.append(int(row[0]))
            
            # people column
            array_people_csv.append(int(row[2]))

    #graph
    x_data_temp_in = []  
    y_data_temp_in = []  
    line_temp_in, = ax1.plot(x_data_temp_in, y_data_temp_in, 'ro-')
    x_data_ipo_temp_in = []  
    y_data_ipo_temp_in = []  
    line_ipo_temp_in, = ax1.plot(x_data_ipo_temp_in, y_data_ipo_temp_in, 'o', color='orange') #point for prediction

    x_data_temp_out = []  
    y_data_temp_out = []  
    line_temp_out, = ax1.plot(x_data_temp_out, y_data_temp_out, 'go-')

    x_data_rew = []  # first 10 points x axes
    y_data_rew = []  # initial data
    line_rew, = ax2.plot(x_data_rew, y_data_rew, 'bo-')
    x_data_ipo_rew = []  
    y_data_ipo_rew = []  
    line_ipo_rew, = ax2.plot(x_data_ipo_rew, y_data_ipo_rew, 'o', color='orange') #point for prediction

    x_data_co2 = []  # first 10 points x axes
    y_data_co2 = []  # initial data
    line_co2, = ax3.plot(x_data_co2, y_data_co2, 'bo-')
    ax3.axhline(y=1000, color='orange', linestyle='--')
    ax3.axhline(y=2500, color='red', linestyle='--') 
    x_data_ipo_co2 = []  
    y_data_ipo_co2 = []  
    line_ipo_co2, = ax3.plot(x_data_ipo_co2, y_data_ipo_co2, 'o',color='orange' ) #point for prediction

    x_data_voc = []  # first 10 points x axes
    y_data_voc = []  # initial data
    line_voc, = ax4.plot(x_data_voc, y_data_voc, 'bo-')
    ax4.axhline(y=600, color='orange', linestyle='--')
    ax4.axhline(y=1500, color='red', linestyle='--')
    x_data_ipo_voc = []  
    y_data_ipo_voc = []  
    line_ipo_voc, = ax4.plot(x_data_ipo_voc, y_data_ipo_voc, 'o', color='orange') #point for prediction 

    x_data_people = array_steps_csv
    y_data_people = array_people_csv
    line_people, = ax5.plot(x_data_people, y_data_people, 'g-')
    vertical_line_people = ax5.axvline(x=0, color='orange', linestyle='--')

    x_data_air_quality_reward = [] 
    y_data_air_quality_reward = []
    line_air_quality_reward, = ax6.plot(x_data_air_quality_reward, y_data_air_quality_reward, 'r-')
    x_data_ipo_air_quality_reward = []  
    y_data_ipo_air_quality_reward = []  
    line_ipo_air_quality_reward, = ax6.plot(x_data_ipo_air_quality_reward, y_data_ipo_air_quality_reward, 'ro') #point for prediction

    x_data_comfort_reward = [] 
    y_data_comfort_reward = []
    line_comfort_reward, = ax6.plot(x_data_comfort_reward, y_data_comfort_reward, 'g-')
    x_data_ipo_comfort_reward = []  
    y_data_ipo_comfort_reward = []  
    line_ipo_comfort_reward, = ax6.plot(x_data_ipo_comfort_reward, y_data_ipo_comfort_reward, 'go') #point for prediction

    x_data_energy_reward = [] 
    y_data_energy_reward = []
    line_energy_reward, = ax6.plot(x_data_energy_reward, y_data_energy_reward, 'b-')
    x_data_ipo_energy_reward = []  
    y_data_ipo_energy_reward = []  
    line_ipo_energy_reward, = ax6.plot(x_data_ipo_energy_reward, y_data_ipo_energy_reward, 'bo') #point for prediction

    # Config graph tot_rewards, reward, actions, people
    ax1.set_title('Temperature')
    ax1.set_xlabel('step')
    ax1.set_ylabel('temp in')
    ax1.legend([line_temp_in,line_temp_out],['temp in','temp out'])

    ax2.set_title('Reward')
    ax2.set_xlabel('step')
    ax2.set_ylabel('reward')
    
    ax3.set_title('Co2')
    ax3.set_xlabel('step')
    ax3.set_ylabel('Co2 level')
    
    ax4.set_title('voc')
    ax4.set_xlabel('step')
    ax4.set_ylabel('voc level')

    ax5.set_title('People')
    ax5.set_xlabel('step')
    ax5.set_ylabel('number')
    ax5.set_xticks(range(0, 121, 20))

    ax6.set_title('Particular rewards')
    ax6.set_xlabel('step')
    ax6.set_ylabel('rewards')
    ax6.legend([line_air_quality_reward, line_comfort_reward, line_energy_reward],['air quality rew','comfort rew','energy rew'])

    #initial graph update
    x_data_temp_in.append(step)
    y_data_temp_in.append(state.temp_in)
    line_temp_in.set_ydata(y_data_temp_in)
    line_temp_in.set_xdata(x_data_temp_in)

    x_data_temp_out.append(step)
    y_data_temp_out.append(state.temp_out)
    line_temp_out.set_ydata(y_data_temp_out)
    line_temp_out.set_xdata(x_data_temp_out)
    ax1.relim()
    ax1.autoscale_view(True, True, True)
    
    x_data_co2.append(step) 
    y_data_co2.append(state.co2)
    line_co2.set_ydata(y_data_co2)
    line_co2.set_xdata(x_data_co2)
    ax3.relim()
    ax3.autoscale_view(True, True, True)
    
    x_data_voc.append(step)
    y_data_voc.append(state.voc)  
    line_voc.set_ydata(y_data_voc)
    line_voc.set_xdata(x_data_voc)
    ax4.relim()
    ax4.autoscale_view(True, True, True)


    if step <= 9:
        ax1.set_xlim(-0.5, 10)
        ax2.set_xlim(-0.5, 10)
        ax3.set_xlim(-0.5, 10)
        ax4.set_xlim(-0.5, 10)
        ax6.set_xlim(-0.5, 10)
    else:
        ax1.set_xlim(step -10, step+1)
        ax2.set_xlim(step -10, step+1)
        ax3.set_xlim(step -10, step+1)
        ax4.set_xlim(step -10, step+1)
        ax6.set_xlim(step -10, step+1)

    plt.draw()
    
    #next step
    step +=1
    
############################Tkinter interface
    OUTPUT_PATH = Path(__file__).parent
    ASSETS_PATH = OUTPUT_PATH / ('assets/frame0')

    #useful for images
    def relative_to_assets(path: str) -> Path:
        return ASSETS_PATH / Path(path)

    #create window
    window = Tk()

    window.geometry("1920x1030")
    window.configure(bg = "#FFFFFF")

    global canvas
    canvas = Canvas(
        window,
        bg = "#FFFFFF",
        height = 1030,
        width = 1920,
        bd = 0,
        highlightthickness = 0,
        relief = "ridge"
    )

    canvas.place(x = 0, y = 0)
    canvas.create_rectangle(
        0.0,
        106.0,
        1920.0,
        228.0,
        fill="#D7D2F5",
        outline="")

    canvas.create_rectangle(
        0.0,
        0.0,
        1920.0,
        132.0,
        fill="#7579F0",
        outline="")

    canvas.create_text(
        700,
        27.0,
        anchor="nw",
        text="SAFEPLACE DASHBOARD",
        fill="#FFFFFF",
        font=("Inter", 40 * -1, "bold"),
    )

    #buttons
    #info button
    button_image_info = PhotoImage(
        file=relative_to_assets("info.png"), master=window)
    button_info = Button(
        image=button_image_info,
        borderwidth=0,
        highlightthickness=0,
        command=on_button_info_click,
        relief="flat",
        master = window
    )
    button_info.place(
        x=0,
        y=228,
        width=60,
        height=60
    )

    #initialize images
    global vent_high_on, vent_low_on
    global wind_on
    global san_on
    vent_high_on = PhotoImage(
        file=relative_to_assets("vent_high_on.png"), master=window)
    vent_low_on = PhotoImage(
        file=relative_to_assets("vent_low_on.png"), master=window)
    wind_on = PhotoImage(
        file=relative_to_assets("wind_on.png"), master=window)
    san_on = PhotoImage(
        file=relative_to_assets("san_on.png"), master=window)
    
    global vent_off
    global label_1
    vent_off = PhotoImage(
        file=relative_to_assets("vent_off.png"), master=window)
    label_1 = tk.Label(window, image=vent_off)
    label_1["background"] = "white"
    label_1.place(x=69, y=297)        

    global wind_off
    global label_2
    wind_off = PhotoImage(
        file=relative_to_assets("wind_off.png"), master=window)
    label_2 = tk.Label(window, image=wind_off)
    label_2["background"] = "white"
    label_2.place(x=225, y=291) 
    
    global san_off
    global label_3
    san_off = PhotoImage(
        file=relative_to_assets("san_off.png"), master=window)
    label_3 = tk.Label(window, image=san_off)
    label_3["background"] = "white"
    label_3.place(x=69, y=412)
    

    #select a mode label
    global label
    label = tk.Label(window, 
                     text="Select a mode: ", 
                     font=("DejaVu Sans", 36 * -1))
    label.place(x=35, y = 155)
    label.config(bg='#D8D3F5')


    ###mode buttons
    #human mode button
    global button_human
    button_image_human = PhotoImage(
        file=relative_to_assets("button_human.png"), master=window)
    button_human = Button(
        image=button_image_human,
        borderwidth=0,
        highlightthickness=0,
        command=on_button_human_click,
        relief="flat",
        master = window
    )
    button_human.place(
        x=335,
        y=155,
        width=353.9110412597656,
        height=44.88607406616211
    )
    #human suggestion button
    global button_human_suggestions
    button_image_human_suggestions = PhotoImage(
        file=relative_to_assets("button_human_suggestions.png"), master=window)
    button_human_suggestions = Button(
        image=button_image_human_suggestions,
        borderwidth=0,
        highlightthickness=0,
        command=on_button_human_suggestions_click,
        relief="flat",
        master = window
    )
    button_human_suggestions.place(
        x=700,
        y=155,
        width=353.9110412597656,
        height=44.88607406616211
    )
    #human ppo button
    global button_ppo_policy
    button_image_ppo_policy = PhotoImage(
        file=relative_to_assets("button_ppo.png"), master=window)
    button_ppo_policy = Button(
        image=button_image_ppo_policy,
        borderwidth=0,
        highlightthickness=0,
        command=on_button_ppo_policy_click,
        relief="flat",
        master = window
    )
    button_ppo_policy.place(
        x=1065,
        y=155,
        width=353.9110412597656,
        height=44.88607406616211
    )

    #ppo score button
    global button_ppo_score
    button_image_ppo_score = PhotoImage(
        file=relative_to_assets("button_ppo_score.png"), master=window)
    button_ppo_score = Button(
        image=button_image_ppo_score,
        borderwidth=0,
        highlightthickness=0,
        command=on_button_ppo_score_click,
        relief="flat",
        master = window
    )
    
    global button_stop
    button_image_stop = PhotoImage(
        file=relative_to_assets("button_stop.png"), master=window)
    button_stop = Button(
        image=button_image_stop,
        borderwidth=0,
        highlightthickness=0,
        command=on_button_stop_click,
        relief="flat",
        master = window
    )

    global button_home
    button_image_home = PhotoImage(
        file=relative_to_assets("button_home.png"), master=window)
    button_home = Button(
        image=button_image_home,
        borderwidth=0,
        highlightthickness=0,
        command=on_button_home_click,
        relief="flat",
        master = window
    )
    
    
    #buttons image
    global button_1, button_2, button_3, button_4, button_5, button_6, button_7, button_8
    global button_image_1, button_image_2, button_image_3, button_image_4, button_image_5, button_image_6, button_image_7, button_image_8
    button_image_1 = PhotoImage(
        file=relative_to_assets("button_1.png"), master=window)
    button_1 = Button(
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        command=on_button1_click,
        relief="flat",
        master = window
    )

    button_image_8 = PhotoImage(
        file=relative_to_assets("button_8.png"), master=window)
    button_8 = Button(
        image=button_image_8,
        borderwidth=0,
        highlightthickness=0,
        command=on_button8_click,
        relief="flat",
        master = window
    )

    button_image_7 = PhotoImage(
        file=relative_to_assets("button_7.png"), master=window)
    button_7 = Button(
        image=button_image_7,
        borderwidth=0,
        highlightthickness=0,
        command=on_button7_click,
        relief="flat",
        master = window
    )
    

    button_image_6 = PhotoImage(
        file=relative_to_assets("button_6.png"), master=window)
    button_6 = Button(
        image=button_image_6,
        borderwidth=0,
        highlightthickness=0,
        command=on_button6_click,
        relief="flat",
        master = window
    )

    button_image_5 = PhotoImage(
        file=relative_to_assets("button_5.png"), master=window)
    button_5 = Button(
        image=button_image_5,
        borderwidth=0,
        highlightthickness=0,
        command=on_button5_click,
        relief="flat",
        master = window
    )

    button_image_4 = PhotoImage(
        file=relative_to_assets("button_4.png"), master=window)
    button_4 = Button(
        image=button_image_4,
        borderwidth=0,
        highlightthickness=0,
        command=on_button4_click,
        relief="flat",
        master = window
    )

    button_image_3 = PhotoImage(
        file=relative_to_assets("button_3.png"), master=window)
    button_3 = Button(
        image=button_image_3,
        borderwidth=0,
        highlightthickness=0,
        command=on_button3_click,
        relief="flat",
        master = window
    )

    button_image_2 = PhotoImage(
        file=relative_to_assets("button_2.png"), master=window)
    button_2 = Button(
        image=button_image_2,
        borderwidth=0,
        highlightthickness=0,
        command=on_button2_click,
        relief="flat",
        master = window
    )


    ##graph visualization
    right_frame = tk.Frame(window, width=600, height=750)
    right_frame.place(x=400, y=230)
    
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.get_tk_widget().grid(row=0, column=0)
    

    #window visualization
    window.protocol("WM_DELETE_WINDOW", close_window)
    window.title("SafePlace")
    window.mainloop()


    
    
    
    
  
    
