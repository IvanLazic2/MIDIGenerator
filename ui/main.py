from tkinter import *
import customtkinter as ctk
from pathlib import Path
import subprocess
import os
import threading
import select
import sys

from managers.process_manager import ProcessManager
from screens.screens import ScreenGenerate, ScreenTrain, ScreenTokenize

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("MIDI Generator")
        self.geometry(f"{1750}x{900}")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.create_sidebar()

        self.process_manager = ProcessManager()

        #self.create_screen_train()
        #self.create_screen_tokenize()
        #self.create_screen_generate()
        self.screen_generate = ScreenGenerate(self, self.process_manager, fg_color="transparent")
        self.screen_tokenize = ScreenTokenize(self, self.process_manager, fg_color="transparent")
        self.screen_train = ScreenTrain(self, self.process_manager, fg_color="transparent")
        

        

        # Default screen
        self.select_screen("generate")


    
    def onclick_button_tokenize(self):
        self.select_screen("tokenize")

    def onclick_button_train(self):
        self.select_screen("train")

    def onclick_button_generatee(self):
        self.select_screen("generate")

    
    def create_sidebar(self):
        sidebar = ctk.CTkFrame(self, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nswe")
        sidebar.grid_rowconfigure(4, weight=1)

        sidebar_title = ctk.CTkLabel(sidebar, text="MIDI Generator", font=("Arial", 20), corner_radius=0)
        sidebar_title.grid(row=0, column=0, sticky="ew", pady=20, padx=20)

        self.button_tokenize = ctk.CTkButton(sidebar, text="Tokenize", corner_radius=0, height=40, border_spacing=10, fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), anchor="w", command=self.onclick_button_tokenize)
        self.button_tokenize.grid(row=1, column=0, sticky="ew")

        self.button_train = ctk.CTkButton(sidebar, text="Train", corner_radius=0, height=40, border_spacing=10, fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), anchor="w", command=self.onclick_button_train)
        self.button_train.grid(row=2, column=0, sticky="ew")

        self.button_generate = ctk.CTkButton(sidebar, text="Generate", corner_radius=0, height=40, border_spacing=10, fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), anchor="w", command=self.onclick_button_generatee)
        self.button_generate.grid(row=3, column=0, sticky="ew")

    def select_screen(self, screen):
        self.button_tokenize.configure(fg_color=("gray75", "gray25") if screen == "tokenize" else "transparent")
        self.button_train.configure(fg_color=("gray75", "gray25") if screen == "train" else "transparent")
        self.button_generate.configure(fg_color=("gray75", "gray25") if screen == "generate" else "transparent")

        if screen == "tokenize":
            self.screen_tokenize.grid(row=0, column=1, sticky="nswe")
            self.screen_train.grid_forget()
            self.screen_generate.grid_forget()
        elif screen == "train":
            self.screen_tokenize.grid_forget()
            self.screen_train.grid(row=0, column=1, sticky="nswe")
            self.screen_generate.grid_forget()
        elif screen == "generate":
            self.screen_tokenize.grid_forget()
            self.screen_train.grid_forget()
            self.screen_generate.grid(row=0, column=1, sticky="nswe")

    def create_process(self, command):
        return subprocess.Popen(
            command,
            cwd="../transformer",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
    
    def display_process_output(self, process, console):
        while True:
            stderr_line = process.stderr.readline()
            if stderr_line:
                self.write_to_console(console, stderr_line)

            if process.poll() is not None:
                break
        
        process.stdout.close()
        process.stderr.close()

    def write_to_console(self, console, line):
        console.configure(state="normal")

        if any(keyword in line for keyword in ["Tokenizing", "Training", "Generating"]):
            console.delete("0.end", "end")
                
        console.insert("end", line)
        console.configure(state="disabled")
        console.yview("end")

    def create_screen_tokenize(self):
        self.screen_tokenize = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        screen_tokenize_label = ctk.CTkLabel(self.screen_tokenize, text="Tokenize", font=("Arial", 20), corner_radius=0)
        screen_tokenize_label.grid(row=0, column=0, sticky="ew", pady=20, padx=20)

        # folder selection -> Dataset directory with midi files

        # button -> Tokenize

    def create_screen_train(self):
        self.screen_train = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        screen_train_label = ctk.CTkLabel(self.screen_train, text="Train", font=("Arial", 20), corner_radius=0)
        screen_train_label.grid(row=0, column=0, sticky="ew", pady=20, padx=20)

        # int -> Epochs

        # folder selection -> Dataset directory

        # button -> Train

    def create_screen_generate(self):
        self.screen_generate = ctk.CTkFrame(self, fg_color="transparent")
        label_screen_generate = ctk.CTkLabel(self.screen_generate, text="Generate", font=("Arial", 20))
        label_screen_generate.grid(row=0, column=0, sticky="w", padx=20, pady=20)

        # directory selection -> Checkpoint directory
        checkpoint_directory = "/home/ivanubuntu/Projects/MusicGenerator/transformer/checkpoints/2024-08-21_15-07-26"
        checkpoint = "checkpoint-005.eqx"
        
        label_checkpoint_directory = ctk.CTkLabel(self.screen_generate, text="Checkpoint directory")
        label_checkpoint_directory.grid(row=1, column=0, sticky="w", padx=(20, 0), pady=5)

        entry_checkpoint_directory = ctk.CTkEntry(self.screen_generate)
        entry_checkpoint_directory.grid(row=2, column=0, sticky="w", padx=(20, 0), pady=5)
        entry_checkpoint_directory.configure(state="disabled")

        def get_checkpoints(directory=""):
            path = Path(directory)
            checkpoints = [f.name for f in path.iterdir() if f.is_dir()]
            checkpoints.sort()
            return checkpoints

        def onclick_button_checkpoint_directory():
            checkpoint_directory = ctk.filedialog.askdirectory()
            entry_checkpoint_directory.configure(state="normal")
            entry_checkpoint_directory.delete(0, 'end')
            entry_checkpoint_directory.insert(0, checkpoint_directory)
            entry_checkpoint_directory.configure(state="disabled")
            show_dropdown_checkpoint()
            dropdown_checkpoint.configure(values=get_checkpoints(checkpoint_directory))

        def onselect_checkpoint(event):
            checkpoint = event

        button_checkpoint_directory = ctk.CTkButton(self.screen_generate, text="Browse", width=5, command=onclick_button_checkpoint_directory)
        button_checkpoint_directory.grid(row=2, column=1, sticky="w", padx=(5, 0), pady=5)

        dropdown_checkpoint = ctk.CTkOptionMenu(self.screen_generate, values=get_checkpoints(), command=onselect_checkpoint)
        def show_dropdown_checkpoint():
            dropdown_checkpoint.grid(row=2, column=2, sticky="w", padx=(5, 0), pady=5)
        dropdown_checkpoint.set("Select checkpoint")
        def hide_dropdown_checkpoint():
            dropdown_checkpoint.grid_forget()
        hide_dropdown_checkpoint()
        
        # dropdown -> Prompt mode (file, unconditional)
        propmpt_mode = "unconditional"

        label_prompt_mode = ctk.CTkLabel(self.screen_generate, text="Prompt mode")
        label_prompt_mode.grid(row=3, column=0, sticky="w", padx=(20, 0), pady=5)

        def onselect_prompt_mode(event):
            propmpt_mode = event


        dropdown_prompt_mode = ctk.CTkOptionMenu(self.screen_generate, values=["file", "unconditional"], command=onselect_prompt_mode)
        dropdown_prompt_mode.grid(row=4, column=0, columnspan=3, sticky="w", padx=20, pady=5)
        dropdown_prompt_mode.set("Select prompt mode")
        
        # file selection -> Midi prompt
        midi_prompt_path = ""
        midi_prompt_slice = None

        label_midi_prompt = ctk.CTkLabel(self.screen_generate, text="Midi prompt")
        label_midi_prompt.grid(row=3, column=1, sticky="w", padx=(20, 0), pady=5)

        entry_midi_prompt = ctk.CTkEntry(self.screen_generate)
        entry_midi_prompt.grid(row=4, column=1, sticky="w", padx=(20, 0), pady=5)
        entry_midi_prompt.configure(state="disabled")


        # float -> Temperature
        temperature = 1.0

        # int -> Max tokens to generate
        max_tokens = 1000

        # ??? folder selection and textbox -> Output file
        output_path = None
        output_generated_only = True

        # button -> Generate
        def onclick_button_generate():
            self.generate(checkpoint_directory, checkpoint, propmpt_mode, midi_prompt_path, temperature, max_tokens, output_path)

        button_generate = ctk.CTkButton(self.screen_generate, text="Generate", command=onclick_button_generate)
        button_generate.grid(row=7, column=0, columnspan=3, sticky="w", padx=20, pady=(80, 20)) 

        self.textbox_console = ctk.CTkTextbox(self.screen_generate, height=250)
        self.textbox_console.grid(row=8, column=0, columnspan=3, sticky="nswe", padx=20, pady=20)
        self.textbox_console.configure(state="disabled")

    def generate(self, checkpoint_directory, checkpoint):
        self.thread_generate = threading.Thread(target=self.subprocess_generate, args=(checkpoint_directory, checkpoint, prompt_mode, midi_prompt_path, midi_prompt_slice, temperature, max_tokens, output_path, output_generated_only))
        self.thread_generate.daemon = True
        self.thread_generate.start()
        #self.thread_generate.join()

    def subprocess_generate(self, checkpoint_directory, checkpoint, prompt_mode, midi_prompt_path, midi_prompt_slice, temperature, max_tokens, output_path, output_generated_only):
        if checkpoint == "" or checkpoint_directory == "":
            return
        
        process = self.create_process(
            [
                "python", "../transformer/generate.py",
                "--checkpoint_directory", checkpoint_directory,
                "--checkpoint", checkpoint,
                "--prompt_mode", prompt_mode,
                "--prompt_midi", midi_prompt_path,
                "--prompt_midi_slice", midi_prompt_slice,
                "--temperature", temperature,
                "--max_to_generate", max_tokens,
                #"--output_file", output_path,
                #"--output_generated_only", output_generated_only
            ]
        )

        self.display_process_output(process, self.textbox_console)



if __name__ == "__main__":
    app = App()

    app.mainloop()