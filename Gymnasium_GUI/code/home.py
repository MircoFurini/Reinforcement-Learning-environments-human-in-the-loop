import tkinter as tk
from pathlib import Path
from tkinter import Tk, Button, Canvas, PhotoImage, Label
from PIL import Image, ImageTk

class Home(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg = "#69D2E7")
        self.controller = controller

        # Title label
        title_label = Label(
            self,
            text="HOME",
            font=("Inter", 40, "bold"),
            fg="#FFFFFF",
            bg="#69D2E7"
        )
        title_label.place(x=880, y=50)

        # Title label
        subtitle_label = Label(
            self,
            text="Select an environment",
            font=("Inter", 30, "bold"),
            fg="#FFFFFF",
            bg="#69D2E7"
        )
        subtitle_label.place(x=760, y=150)

        button_1 = Button(
            self,
            text="FROZEN LAKE",
            font=("Inter", 15, "bold"),
            bg="#F38630",
            fg="#FFFFFF",
            command=lambda: controller.show_frame("FrozenLakePage")
        )
        button_1.place(x=850, y=306.0, width=240.0, height=70.0)

        button_2 = Button(
            self,
            bg="#F38630",
            fg="#FFFFFF",
            font=("Inter", 15, "bold"),
            text="CLIFF WALKING",
            command=lambda: controller.show_frame("CliffWalkingPage")
        )
        button_2.place(x=850, y=419.0, width=240.0, height=70.0)
        

        button_3 = Button(
            self,
            text="MOUNTAIN CAR",
            font=("Inter", 15, "bold"),
            bg="#F38630",
            fg="#FFFFFF",
            command=lambda: controller.show_frame("MountainCarPage")
        )
        button_3.place( x=850, y=645.0, width=240.0, height=70.0)
        
        button_4 = Button(
            self,
            text="CAR RACING",
            font=("Inter", 15, "bold"),
            bg="#F38630",
            fg="#FFFFFF",
            command=lambda: controller.show_frame("CarRacingPage")
        )
        button_4.place(x=850, y=755.0, width=240.0, height=70.0)

        button_5 = Button(
            self,
            text="TAXI",
            font=("Inter", 15, "bold"),
            bg="#F38630",
            fg="#FFFFFF",
            command=lambda: controller.show_frame("TaxiPage")
        )
        button_5.place( x=850, y=532.0, width=240.0, height=70.0)
        

        
       
    

