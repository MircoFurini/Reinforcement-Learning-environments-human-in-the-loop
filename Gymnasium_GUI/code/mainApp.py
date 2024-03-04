import tkinter as tk
from tkinter import font as tkfont
from home import Home
from frozen_lake_page import FrozenLakePage
from mountain_car_page import MountainCarPage
from car_racing_page import CarRacingPage
from cliff_walking_page import CliffWalkingPage
from taxi_page import TaxiPage
from screeninfo import get_monitors
from tkinter import messagebox

def get_screen_resolution():
        
        monitors = get_monitors()

        if monitors:
            
            monitor = monitors[0]
            width = monitor.width
            height = monitor.height
            return width, height
        else:
            
            return 800, 600  
        
get_screen_resolution

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        app.destroy()
        exit()

class MainApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")

        self.geometry("1900x1000")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (Home, FrozenLakePage, MountainCarPage, CarRacingPage, CliffWalkingPage, TaxiPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("Home")
        
    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

if __name__ == "__main__":
    app = MainApp()
    app.protocol("WM_DELETE_WINDOW", on_closing)
    #app.protocol("WM_DELETE_WINDOW", lambda: app.destroy())
    app.mainloop()
    