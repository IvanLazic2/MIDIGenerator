import datetime
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Optional
from matplotlib import pyplot as plt

from model import MIDIGeneratorModel
from utils import midi_to_notes, key_order, seq_length, vocab_size, data_dir

from generate import main as generate

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
    seq_length = seq_length+1

    windows = dataset.window(seq_length, shift=1, stride=1,
                                drop_remainder=True)

    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    def scale_pitch(x):
      x = x/[vocab_size,1.0,1.0]
      return x

    def split_labels(sequences):
      inputs = sequences[:-1]
      labels_dense = sequences[-1]
      labels = {key:labels_dense[i] for i,key in enumerate(key_order)}

      return scale_pitch(inputs), labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

def save_validation_graph(loss, pitch_accuracy, step_mse, duration_mse):
    fig, ax = plt.subplots()
    ax.plot(loss, label="Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax2 = ax.twinx()
    ax2.plot(pitch_accuracy, color="orange", label="Accuracy")
    ax2.set_ylabel("Accuracy (%)")
    fig.legend()
    fig.savefig(f"metrics.png")

def train(model, train_ds, validation_ds):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            verbose=1,
            restore_best_weights=True),
    ]

    epochs = 20

    history = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=validation_ds,
    )

    print("History keys:", history.history.keys())
    #save_validation_graph(history.history['loss'], history.history['val_pitch_accuracy'], history.history['val_step_mse'], history.history['val_duration_mse'])
    

def main():
    notes_ds, n_notes = create_dataset()

    seq_ds = create_sequences(notes_ds, seq_length, vocab_size)
    seq_ds.element_spec

    batch_size = 64
    buffer_size = n_notes - seq_length

    seq_ds = seq_ds.shuffle(buffer_size)

    train_size = int(0.9 * buffer_size)
    val_size = buffer_size - train_size

    train_ds = (seq_ds
                .take(train_size)
                .batch(batch_size, drop_remainder=True)
                .cache()
                .prefetch(tf.data.experimental.AUTOTUNE))

    validation_ds = (seq_ds
                     .skip(train_size)
                     .take(val_size)
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

    model = MIDIGeneratorModel(inputs = inputs, outputs = outputs)

    model.compile_model()

    model.evaluate(train_ds, return_dict=True)

    train(model, train_ds, validation_ds)

    model.save('model.keras')

    print('Training complete')

if __name__ == '__main__':
    main()
