import tkinter as tk
import customtkinter as ctk

from layout_items.text import *
from layout_items.options import *
from layout_items.console import *

TEMP_PROJ_DIR = "../transformer"

class Screen(ctk.CTkFrame):
    def __init__(self, master, title, process_manager, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        self.process_manager = process_manager

        self.title = screen_title(self, "Generate")

        self.frame = ctk.CTkFrame(self, fg_color="transparent")
        self.frame.grid(row=1, column=0, sticky="nw", padx=0, pady=0)


class ScreenGenerate(Screen):
    def __init__(self, master, process_manager, *args, **kwargs):
        super().__init__(master, "Generate", process_manager, *args, **kwargs)

        options_frame = ctk.CTkFrame(self.frame)
        options_frame.grid(row=0, column=0, sticky="nw", padx=(20, 0), pady=0)

        self.checkpoint_directory_browser = CheckpointBrowser(options_frame, default="/home/ivanubuntu/Projects/MusicGenerator/transformer/checkpoints/2024-08-21_15-07-26", default_checkpoint="checkpoint-005.eqx")
        self.checkpoint_directory_browser.grid(row=0, column=0, sticky="w", padx=20, pady=(20, 20))

        self.prompt_mode_dropdown = MidiPromptDropdown(options_frame, default="unconditional", default_midi_path="")
        self.prompt_mode_dropdown.grid(row=1, column=0, sticky="w", padx=20, pady=(0, 20))

        self.temperature_input = DecimalInput(options_frame, label="Temperature", default=1.0)
        self.temperature_input.grid(row=2, column=0, sticky="w", padx=20, pady=(0, 20))

        self.max_tokens_input = IntInput(options_frame, label="Max tokens", default=1000)
        self.max_tokens_input.grid(row=3, column=0, sticky="w", padx=20, pady=(0, 20))

        self.output_browser = OutputBrowser(options_frame, default="", default_output_name="output.mid")
        self.output_browser.grid(row=4, column=0, sticky="w", padx=20, pady=(0, 20))

        self.output_only_generated_checkbox = Checkbox(options_frame, label="Output only generated", default=False)
        self.output_only_generated_checkbox.grid(row=5, column=0, sticky="w", padx=20, pady=(0, 20))
        
        console_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        console_frame.grid(row=0, column=1, sticky="nw", padx=(20, 0), pady=0)

        self.console = Console(console_frame, action_name="Generate", action_callback=self.generate)
        self.console.grid(row=0, column=0, sticky="nw", padx=0, pady=0)

    def generate(self):
        self.checkpoint_directory = self.checkpoint_directory_browser.selected
        self.checkpoint = self.checkpoint_directory_browser.checkpoint
        self.prompt_mode = self.prompt_mode_dropdown.selected
        self.prompt_midi = self.prompt_mode_dropdown.midi_browser.selected
        self.temperature = self.temperature_input.value.get()
        self.max_tokens = self.max_tokens_input.value.get()
        self.output = self.output_browser.selected
        self.output_only_generated = self.output_only_generated_checkbox.value

        if self.checkpoint_directory == "" or self.checkpoint == "":
            return

        if self.prompt_mode == "file" and self.prompt_midi == "":
            return

        command = [
            "python", f"{TEMP_PROJ_DIR}/generate.py",
            "--checkpoint_directory", self.checkpoint_directory,
            "--checkpoint", self.checkpoint,
            "--prompt_mode", self.prompt_mode,
            "--prompt_midi", self.prompt_midi,
            #"--prompt_midi_slice", None, # AAAAAAAAAAAAAAAAAAAA
            "--temperature", self.temperature,
            "--max_to_generate", self.max_tokens,
            #"--output_file", self.output,
            #"--output_generated_only", self.output_only_generated
        ]

        self.process_manager.start_process(self.console, command, TEMP_PROJ_DIR)


class ScreenTokenize(Screen):
    def __init__(self, master, process_manager, *args, **kwargs):
        super().__init__(master, "Tokenize", process_manager, *args, **kwargs)

        options_frame = ctk.CTkFrame(self.frame)
        options_frame.grid(row=0, column=0, sticky="nw", padx=(20, 0), pady=0)

        # folder selection -> Dataset directory with midi files
        self.dataset_directory_browser = Browser(options_frame, label="MIDI Dataset Directory", default="/home/ivanubuntu/Projects/MusicGenerator/datasets/GiantMIDI-PIano_micro/midis", browser_type="directory")
        self.dataset_directory_browser.grid(row=0, column=0, sticky="w", padx=20, pady=(20, 20))

        # output -> Tokenized dataset directory
        self.output_directory_browser = Browser(options_frame, label="Tokenized Dataset Directory", default="/home/ivanubuntu/Projects/MusicGenerator/transformer/dataset_tokenized", browser_type="directory")
        self.output_directory_browser.grid(row=1, column=0, sticky="w", padx=20, pady=(0, 20))

        # output tokenizer.json -> Tokenizer json file
        self.tokenizer_json_browser = Browser(options_frame, label="tokenizer.json Directory", default="/home/ivanubuntu/Projects/MusicGenerator/transformer", browser_type="directory")
        self.tokenizer_json_browser.grid(row=2, column=0, sticky="w", padx=20, pady=(0, 20))

        console_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        console_frame.grid(row=0, column=1, sticky="nw", padx=(20, 0), pady=0)

        self.console = Console(console_frame, action_name="Tokenize", action_callback=self.tokenize)
        self.console.grid(row=0, column=0, sticky="nw", padx=0, pady=0)

    def tokenize(self):
        self.dataset_directory = self.dataset_directory_browser.selected
        self.output_directory = self.output_directory_browser.selected
        self.tokenizer_json = self.tokenizer_json_browser.selected

        if self.dataset_directory == "" or self.output_directory == "" or self.tokenizer_json == "":
            return
        
        command = [
            "python", "-m" "data.tokenize_dataset",
            "--midis_dir", self.dataset_directory,
            "--out_dir", self.output_directory,
            "--out_tokenizer", self.tokenizer_json + "/tokenizer.json"
        ]

        self.process_manager.start_process(self.console, command, TEMP_PROJ_DIR)





class ScreenTrain(Screen):
    def __init__(self, master, process_manager, *args, **kwargs):
        super().__init__(master, "Train", process_manager, *args, **kwargs)

        options_frame = ctk.CTkFrame(self.frame)
        options_frame.grid(row=0, column=0, sticky="nw", padx=(20, 0), pady=0)

        # int -> Epochs
        self.epochs_input = IntInput(options_frame, label="Epochs", default=5)
        self.epochs_input.grid(row=0, column=0, sticky="w", padx=20, pady=(20, 20))

        # folder selection -> Dataset directory
        self.dataset_directory_browser = Browser(options_frame, label="Tokenized Dataset", default="/home/ivanubuntu/Projects/MusicGenerator/transformer/dataset_tokenized", browser_type="directory")
        self.dataset_directory_browser.grid(row=1, column=0, sticky="w", padx=20, pady=(0, 20)) 

        console_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        console_frame.grid(row=0, column=1, sticky="nw", padx=(20, 0), pady=0)

        self.console = Console(console_frame, action_name="Train", action_callback=self.train)
        self.console.grid(row=0, column=0, sticky="nw", padx=0, pady=0)

    def train(self):
        self.epochs = self.epochs_input.value.get()
        self.dataset_directory = self.dataset_directory_browser.selected

        if self.epochs == "" or self.dataset_directory == "":
            return

        command = [
            "python", f"{TEMP_PROJ_DIR}/train.py",
            "--epochs", self.epochs,
            "--dataset", self.dataset_directory
        ]

        self.process_manager.start_process(self.console, command, TEMP_PROJ_DIR)

        