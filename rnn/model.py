import keras
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable

from utils import midi_to_notes, key_order, seq_length, vocab_size

learning_rate = 0.005

@register_keras_serializable(package='Custom', name='mse_with_positive_pressure')
def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
    mse = (y_true - y_pred) ** 2
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + positive_pressure)

@register_keras_serializable(package='Custom', name='MIDIGeneratorModel')
class MIDIGeneratorModel(tf.keras.Model):
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

    def compile_model(self):
        loss = {
          'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
              from_logits=True),
          'step': mse_with_positive_pressure,
          'duration': mse_with_positive_pressure,
        }

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        super().compile(
            loss=loss,
            loss_weights={
            'pitch': 0.05,
            'step': 1.0,
            'duration': 1.0,
            },
            optimizer=optimizer,
            metrics={
                'pitch': tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                'step': tf.keras.metrics.MeanSquaredError(name='mse'),
                'duration': tf.keras.metrics.MeanSquaredError(name='mse'),
            }
        )