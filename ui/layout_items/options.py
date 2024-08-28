from tkinter import *
import customtkinter as ctk

from pathlib import Path

from layout_items.text import *
from layout_items.buttons import *

class Option(ctk.CTkFrame):
    def __init__(self, master, label, default, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        self.label = label
        self.default = default

        self.frame = ctk.CTkFrame(self, fg_color="transparent")
        self.frame.grid(row=0, column=0, padx=20, pady=20, sticky="w")

        self.label = option_label(self.frame, label, row=0, column=0, padx=0, pady=(0, 10))

class Browser(Option):
    def __init__(self, master, label, default, browser_type, *args, **kwargs):
        super().__init__(master, label, default, *args, **kwargs)

        self.browser_type = browser_type

        self.selected = self.default

        self.entry = ctk.CTkEntry(self.frame, width=400)
        self.entry.grid(row=1, column=0, padx=(0, 10), sticky="w")
        self.entry.insert(0, self.selected)
        self.entry.xview("end")
        self.entry.configure(state="disabled")

        self.button = BrowseButton(self.frame, self.onbrowse)
        self.button.grid(row=1, column=1, sticky="w")

    def onbrowse(self):
        if self.browser_type == "file":
            self.selected = ctk.filedialog.askopenfilename()
        elif self.browser_type == "directory":
            self.selected = ctk.filedialog.askdirectory()

        self.entry.configure(state="normal")
        self.entry.delete(0, "end")
        self.entry.insert(0, self.selected)
        self.entry.xview("end")
        self.entry.configure(state="disabled")
        
class CheckpointBrowser(Browser):
    def __init__(self, master, default, default_checkpoint, *args, **kwargs):
        super().__init__(master, label="Checkpoint directory", default=default, browser_type="directory", *args, **kwargs)

        self.checkpoint = default_checkpoint

        self.checkpoint_frame = ctk.CTkFrame(self, fg_color="transparent")

        self.checkpoint_label = option_label(self.checkpoint_frame, "Checkpoint", row=0, column=0, padx=0, pady=(0, 10))

        self.dropdown = ctk.CTkOptionMenu(self.checkpoint_frame, values=self.get_checkpoints(), command=self.onselect)
        self.dropdown.grid(row=1, column=0, sticky="w")
        self.dropdown.set("Select checkpoint")

    def onbrowse(self):
        super().onbrowse()
        self.show_dropdown()
        self.dropdown.configure(values=self.get_checkpoints())

    def onselect(self, event):
        self.checkpoint = event

    def show_dropdown(self):
        self.checkpoint_frame.grid(row=1, column=0, padx=20, pady=20, sticky="w")

    def hide_dropdown(self):
        self.checkpoint_frame.grid_forget()

    def get_checkpoints(self):
            path = Path(self.selected)
            checkpoints = [f.name for f in path.iterdir() if f.is_dir()]
            checkpoints.sort()
            return checkpoints
        
class Dropdown(Option):
    def __init__(self, master, label, default, values, *args, **kwargs):
        super().__init__(master, label, default, *args, **kwargs)

        self.selected = self.default

        self.dropdown = ctk.CTkOptionMenu(self.frame, values=values, command=self.onselect)
        self.dropdown.grid(row=1, column=0, sticky="w")
        self.dropdown.set("Select")

    def onselect(self, event):
        self.selected = event

class MidiPromptDropdown(Dropdown):
    def __init__(self, master, default, default_midi_path, *args, **kwargs):
        super().__init__(master, label="Prompt mode", default=default, values=["unconditional", "file"], *args, **kwargs)

        self.midi_browser = Browser(self, label="Midi prompt", default=default_midi_path, browser_type="file", fg_color="transparent")
        
    def onselect(self, event):
        super().onselect(event)
        if event == "file":
            self.show_browser()
        else:
            self.hide_browser()

    def show_browser(self):
        self.midi_browser.grid(row=1, column=0, sticky="w")

    def hide_browser(self):
        self.midi_browser.grid_forget()

class DecimalInput(Option):
    def __init__(self, master, label, default, *args, **kwargs):
        super().__init__(master, label, default, *args, **kwargs)

        self.value = StringVar(self)
        self.value.set(str(self.default))

        self.entry = ctk.CTkEntry(self.frame, width=100, textvariable=self.value, validatecommand=self.validate_input)  
        self.entry.grid(row=1, column=0, sticky="w")

    def validate_input(self, new_value):
        if new_value == "":
            return True

        try:
            float(new_value)
            return True
        except ValueError:
            return False
        
        return False

class IntInput(Option):
    def __init__(self, master, label, default, *args, **kwargs):
        super().__init__(master, label, default, *args, **kwargs)

        self.value = StringVar(self)
        self.value.set(str(self.default))

        self.entry = ctk.CTkEntry(self.frame, width=100, textvariable=self.value, validatecommand=self.validate_input)
        self.entry.grid(row=1, column=0, sticky="w")

    def validate_input(self, new_value):
        if new_value == "":
            return True

        try:
            int(new_value)
            return True
        except ValueError:
            return False
        
        return False

class OutputBrowser(Browser):
    def __init__(self, master, default, default_output_name, *args, **kwargs):
        super().__init__(master, label="Output folder", default=default, browser_type="directory", *args, **kwargs)

        self.output_name = StringVar(self)
        self.output_name.set(default_output_name)

        self.output_frame = ctk.CTkFrame(self, fg_color="transparent")

        self.output_label = option_label(self.output_frame, "File name", row=0, column=0, padx=0, pady=(0, 10))

        self.output_entry = ctk.CTkEntry(self.output_frame, width=200, textvariable=self.output_name)
        self.output_entry.grid(row=1, column=0, padx=(0, 10), sticky="w")

    def onbrowse(self):
        super().onbrowse()
        self.show_file_name()

    def show_file_name(self):
        self.output_frame.grid(row=1, column=0, padx=20, pady=20, sticky="w")

    def hide_file_name(self):
        self.output_frame.grid_forget()

class Checkbox(Option):
    def __init__(self, master, label, default, *args, **kwargs):
        super().__init__(master, label, default, *args, **kwargs)

        self.value = BooleanVar(self)
        self.value.set(self.default)

        self.checkbox = ctk.CTkCheckBox(self.frame, text="", variable=self.value)
        self.checkbox.grid(row=0, column=1, padx=(10, 0), pady=(0, 10), sticky="wns")

    

