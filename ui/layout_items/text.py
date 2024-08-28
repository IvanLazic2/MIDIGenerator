from tkinter import *
import customtkinter as ctk

FONT = "Arial"

def screen_title(master, text):
    title = ctk.CTkLabel(master, text=text, font=(FONT, 20))
    title.grid(row=0, column=0, padx=20, pady=20, sticky="w")
    return title

def option_label(master, text, row=0, column=0, padx=10, pady=10):
    label = ctk.CTkLabel(master, text=text)
    label.grid(row=row, column=column, padx=padx, pady=pady, sticky="nw")
    return label