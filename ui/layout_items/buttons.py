from tkinter import *
import customtkinter as ctk

from layout_items.text import *

class MainActionButton(ctk.CTkButton):
    def __init__(self, master, text, callback, *args, **kwargs):
        super().__init__(master, text=text, font=(FONT, 20), width=200, height=50, command=callback, *args, **kwargs)

class BrowseButton(ctk.CTkButton):
    def __init__(self, master, callback, *args, **kwargs):
        super().__init__(master, text="Browse", width=5, command=callback, *args, **kwargs)

    

