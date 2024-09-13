import datetime
import glob
import numpy as np
import pathlib
import pandas as pd
import tensorflow as tf

from typing import Optional

from model import MIDIGeneratorModel
from utils import midi_to_notes, key_order, seq_length, vocab_size

data_dir = pathlib.Path('/home/ivanubuntu/Projects/MIDIGenerator/datasets/Maestro v3/extracted_files_maestro_single')


def create_dataset():
    filenames = glob.glob(str(data_dir/'*.mid*'))
    if len(filenames) == 0:
        filenames = glob.glob(str(data_dir/'*.midi*'))

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



def train(model, train_ds):
    checkpoint_dir = ''#datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    #pathlib.Path('./checkpoints/' + checkpoint_dir).mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./checkpoints/' + checkpoint_dir + 'checkpoint_{epoch}.weights.h5',
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


    inputs = tf.keras.Input(input_shape)
    x = tf.keras.layers.LSTM(128)(inputs)

    outputs = {
      'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
      'step': tf.keras.layers.Dense(1, name='step')(x),
      'duration': tf.keras.layers.Dense(1, name='duration')(x),
    }

    model = MIDIGeneratorModel(inputs=inputs, outputs=outputs)

    model.compile()

    model.evaluate(train_ds, return_dict=True)

    train(model, train_ds)

    print('Training complete')

if __name__ == '__main__':
    main()
