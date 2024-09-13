import collections
import datetime
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import tensorflow as tf


from typing import Optional

data_dir = pathlib.Path('../datasets/Maestro v3/extracted_files_maestro_single')
key_order = ['pitch', 'step', 'duration']

def midi_to_notes(midi_file: str) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    # Sort the notes by start time
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

def create_dataset():
    filenames = glob.glob(str(data_dir/'**/*.mid*'))
    print('Number of files:', len(filenames))

    num_files = 5
    all_notes = []
    for f in filenames[:num_files]:
        notes = midi_to_notes(f)
        all_notes.append(notes)

    all_notes = pd.concat(all_notes)

    n_notes = len(all_notes)
    print('Number of notes parsed:', n_notes)

    train_notes = np.stack([all_notes[key] for key in key_order], axis=1)

    return tf.data.Dataset.from_tensor_slices(train_notes), n_notes

def create_sequences(
    dataset: tf.data.Dataset, 
    seq_length: int,
    vocab_size = 128,
) -> tf.data.Dataset:
    """Returns TF Dataset of sequence and label examples."""
    seq_length = seq_length+1

    # Take 1 extra for the labels
    windows = dataset.window(seq_length, shift=1, stride=1,
                                drop_remainder=True)

    # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)
    
    # Normalize note pitch
    def scale_pitch(x):
      x = x/[vocab_size,1.0,1.0]
      return x

    # Split the labels
    def split_labels(sequences):
      inputs = sequences[:-1]
      labels_dense = sequences[-1]
      labels = {key:labels_dense[i] for i,key in enumerate(key_order)}

      return scale_pitch(inputs), labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
    mse = (y_true - y_pred) ** 2
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + positive_pressure)

def train(model, train_ds):
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./checkpoints/checkpoint_{epoch}',
            save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            verbose=1,
            restore_best_weights=True),
    ]

    epochs = 50

    history = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

def main():
    notes_ds, n_notes = create_dataset()

    seq_length = 25
    vocab_size = 128
    seq_ds = create_sequences(notes_ds, seq_length, vocab_size)
    seq_ds.element_spec

    batch_size = 64
    buffer_size = n_notes - seq_length  # the number of items in the dataset
    train_ds = (seq_ds
                .shuffle(buffer_size)
                .batch(batch_size, drop_remainder=True)
                .cache()
                .prefetch(tf.data.experimental.AUTOTUNE))

    input_shape = (seq_length, 3)
    learning_rate = 0.005

    inputs = tf.keras.Input(input_shape)
    x = tf.keras.layers.LSTM(128)(inputs)

    outputs = {
      'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
      'step': tf.keras.layers.Dense(1, name='step')(x),
      'duration': tf.keras.layers.Dense(1, name='duration')(x),
    }

    model = tf.keras.Model(inputs, outputs)

    loss = {
          'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
              from_logits=True),
          'step': mse_with_positive_pressure,
          'duration': mse_with_positive_pressure,
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss=loss,
        loss_weights={
            'pitch': 0.05,
            'step': 1.0,
            'duration':1.0,
        },
        optimizer=optimizer,
    )

    model.evaluate(train_ds, return_dict=True)

    train(model, train_ds)

    print('Training complete')


