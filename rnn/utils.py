import collections
import pathlib
import pandas as pd
import pretty_midi
import numpy as np

data_dir = pathlib.Path('/home/ivanubuntu/Projects/MIDIGenerator/datasets/Maestro v3/extracted_files_maestro_single')
#data_dir = pathlib.Path('/home/ivanubuntu/Projects/MIDIGenerator/datasets/POP909-Dataset/extracted_files_pop909_single')
key_order = ['pitch', 'step', 'duration']
seq_length = 25
vocab_size = 128

def midi_to_notes(midi_file: str) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

def notes_to_midi(
      notes: pd.DataFrame,
      out_file: str, 
      instrument_name: str,
      velocity: int = 100,
    ) -> pretty_midi.PrettyMIDI:

      pm = pretty_midi.PrettyMIDI()
      instrument = pretty_midi.Instrument(
          program=pretty_midi.instrument_name_to_program(
              instrument_name))

      prev_start = 0
      for i, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note['pitch']),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

      pm.instruments.append(instrument)
      pm.write(out_file)
      return pm