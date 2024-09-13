import tensorflow as tf

learning_rate = 0.005

class MIDIGeneratorModel(tf.keras.Model):
    def __init__(self, inputs, outputs):
        super().__init__(inputs, outputs)

    def mse_with_positive_pressure(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        mse = (y_true - y_pred) ** 2
        positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
        return tf.reduce_mean(mse + positive_pressure)
    
    def compile(self):
        loss = {
          'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
              from_logits=True),
          'step': self.mse_with_positive_pressure,
          'duration': self.mse_with_positive_pressure,
        }

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        super().compile(
            loss=loss,
            loss_weights={
                'pitch': 0.05,
                'step': 1.0,
                'duration':1.0,
            },
            optimizer=optimizer,
        )   