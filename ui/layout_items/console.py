import tkinter as tk
import customtkinter as ctk
import os
import select
import sys
import re


from layout_items.buttons import *
from layout_items.text import *
class Console(ctk.CTkFrame):
    def __init__(self, master, action_name, action_callback, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        self.process = None

        self.command = ""
        self.working_directory = ""
        self.action_callback = action_callback

        self.show_console = ctk.BooleanVar()
        self.show_console.set(True)

        self.frame = ctk.CTkFrame(self, fg_color="transparent")
        self.frame.grid(row=0, column=0, padx=20, pady=0)

        self.console_window = ConsoleWindow(self.frame)
        self.console_window.grid(row=0, column=0, sticky="nw")
        self.toggle_console()

        self.console_buttons_frame = ctk.CTkFrame(self.frame)
        self.console_buttons_frame.grid(row=1, column=0, sticky="w", padx=0, pady=20)

        self.action_button = MainActionButton(self.console_buttons_frame, text=action_name, callback=action_callback)
        self.action_button.grid(row=1, column=0, sticky="w", padx=0, pady=(0, 0))

        self.show_console_frame = ctk.CTkFrame(self.console_buttons_frame, fg_color="transparent")
        self.show_console_frame.grid(row=1, column=1, sticky="w", padx=0, pady=0)

        self.show_console_label = option_label(self.show_console_frame, "Show console", row=0, column=0, padx=(20, 0), pady=0)
        self.show_checkbox = ctk.CTkCheckBox(self.show_console_frame, text="", variable=self.show_console, command=self.toggle_console)
        self.show_checkbox.grid(row=0, column=1, padx=(10, 0), pady=0, sticky="w")

        self.progressbar_frame = ctk.CTkFrame(self.console_buttons_frame, fg_color="transparent")

        self.progress_title = ctk.CTkLabel(self.progressbar_frame, text="", font=("Arial", 12))
        

        self.progress_percentage = ctk.CTkLabel(self.progressbar_frame, text="0%", font=("Arial", 12))
        self.progress_percentage.grid(row=0, column=0, sticky="w", padx=0, pady=0)

        self.progressbar = ctk.CTkProgressBar(self.progressbar_frame, orientation="horizontal")
        self.progressbar.grid(row=0, column=1, sticky="w", padx=10, pady=0)
        self.progressbar.set(0)

        self.progress_remaining = ctk.CTkLabel(self.progressbar_frame, text="00:00", font=("Arial", 12))
        self.progress_remaining.grid(row=0, column=2, sticky="w", padx=0, pady=0)

        self.kill_button = ctk.CTkButton(self.progressbar_frame, text="✖", width=30, height=30, font=("Arial", 20), fg_color="orange", hover_color="red")
        self.kill_button.grid(row=0, column=3, sticky="w", padx=(20, 0), pady=0)

    def toggle_console(self):
        if self.show_console.get():
            self.console_window.grid(row=0, column=0, sticky="nw", padx=0, pady=0)
        else:
            self.console_window.grid_forget()

    def display_progress_info(self, line):
        progress_info = self.console_window.get_tqdm_info(line)
        if progress_info:
            if progress_info["title"] == "":
                self.progress_title.grid_forget()
            else:
                self.progress_title.grid(row=1, column=1, sticky="we", padx=0, pady=0)

            title = re.sub(r"\(.*\)", "", progress_info["title"])[:20]
            self.progress_title.configure(text=title.strip())
            self.progress_percentage.configure(text=f"{progress_info['percentage']}%")
            self.progressbar.set(int(progress_info["percentage"]) / 100)
            self.progress_remaining.configure(text=progress_info["remaining"])

    def reset_progress(self):
        self.progressbar.set(0)
        self.progress_title.configure(text="")
        self.progress_percentage.configure(text="0%")
        self.progress_remaining.configure(text="00:00")
        self.progressbar.configure(progress_color=ctk.ThemeManager.theme["CTkProgressBar"]["progress_color"])

    def show_progress(self):
        self.reset_progress()
        self.progressbar_frame.grid(row=1, column=2, sticky="w", padx=20, pady=0)

    def hide_progress(self):
        self.progressbar_frame.grid_forget()

class ConsoleWindow(ctk.CTkFrame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        self.textbox = ctk.CTkTextbox(self, width=765, height=625)
        self.textbox.grid(row=0, column=0, columnspan=3, sticky="nswe", pady=(20, 0))
        self.textbox.configure(state="disabled")

    def get_tqdm_info(self, line):
        # extract a percentage from tqdm output with regex
        regex = r"(.*): *(\d+)%.*(\d+/\d+) +\[(\d+:\d+)<(\d+:\d+), +(\d+.\d+.*\/s)\]"
        result = re.search(regex, line)
        if result:
            info = {
                "title": result.group(1),
                "percentage": result.group(2),
                "progress": result.group(3),
                "elapsed": result.group(4),
                "remaining": result.group(5),
                "rate": result.group(6)
            }
            return info

    def write(self, text):
        self.textbox.configure(state="normal")
        self.textbox.insert("end", text)
        self.textbox.see("end")
        self.textbox.configure(state="disabled")

    def delete_line(self):
        self.textbox.configure(state="normal")
        self.textbox.delete("0.end", "end")
        self.textbox.configure(state="disabled")

    def clear(self):
        self.textbox.configure(state="normal")
        self.textbox.delete("0.0", "end")
        self.textbox.configure(state="disabled")