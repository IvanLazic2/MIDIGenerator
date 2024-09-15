import datetime
import glob
import numpy as np
import pandas as pd
import pretty_midi
import keras
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope

from model import MIDIGeneratorModel
from utils import midi_to_notes, notes_to_midi, key_order, seq_length, vocab_size, data_dir

def predict_next_note(
      notes: np.ndarray, 
      model: tf.keras.Model, 
      temperature: float = 1.0) -> tuple[int, float, float]:
    """Generates a note as a tuple of (pitch, step, duration), using a trained sequence model."""

    assert temperature > 0

    inputs = tf.expand_dims(notes, 0)

    predictions = model.predict(inputs)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']
    
    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)

    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)

    return int(pitch), float(step), float(duration)

def generate(model, raw_notes):
    temperature = 2.0
    num_predictions = 120

    sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

    input_notes = (
        sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

    generated_notes = []
    prev_start = 0
    for _ in range(num_predictions):
        pitch, step, duration = predict_next_note(input_notes, model, temperature)
        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, start, end))
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
        prev_start = start

    generated_notes = pd.DataFrame(generated_notes, columns=(*key_order, 'start', 'end'))
    
    return generated_notes

def main(model):
    if model is None:
        with custom_object_scope({'MIDIGeneratorModel': MIDIGeneratorModel}):
            model = keras.models.load_model('model.keras')

    filenames = glob.glob(str(data_dir/'*.mid*'))

    sample_file = filenames[1]
    sample_pm = pretty_midi.PrettyMIDI(sample_file)
    sample_out_file = f"current_sample_midi.mid"
    sample_pm.write(sample_out_file)

    pm = pretty_midi.PrettyMIDI(sample_file)
    instrument = pm.instruments[0]
    instrument_name = pretty_midi.program_to_instrument_name(instrument.program)

    raw_notes = midi_to_notes(sample_file)

    generated_notes = generate(model, raw_notes)

    out_file = f"generated_midis/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mid"
    out_pm = notes_to_midi(generated_notes, out_file = out_file, instrument_name = instrument_name)
    out_pm.write(out_file)


if __name__ == '__main__':
    main(None)