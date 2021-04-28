import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout 
from tensorflow.keras import regularizers

tf.random.set_seed(1)

class DropConnectModel(keras.Model):

    def __init__(self, num_classes, prob=0.1, use_dropConnect=True):
        super(DropConnectModel, self).__init__(name='mlp')
        self.use_dropConnect = use_dropConnect
        self.num_classes = num_classes

        if self.use_dropConnect:
          self.dense_drop_connect = DropConnect(prob=prob,units=64, activation='relu')
          self.dense_drop_connect2 = DropConnect(prob=prob,units=64, activation='relu')
          # self.dense_drop_connect2 = DropConnect(prob=prob,units=128, activation='relu')
          # self.dense_drop_connect2 = DropConnect(prob=prob,units=128, activation='relu')
          # self.dense_drop_connect2 = DropConnect(prob=prob,units=128, activation='relu')
        else:
          self.dense2 = Dense(32, activation='relu')
          self.dense3 = Dense(32, activation='relu')
          self.dropout = Dropout(prob)
        self.dense_out = Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
      # x = keras.layers.Flatten()(inputs)
      x = inputs
      if self.use_dropConnect:
        x = self.dense_drop_connect(x, training=training)
        x = self.dense_drop_connect2(x, training=training)
      else:
        x = self.dense2(x)
        x = self.dropout(x, training=training)
      x = self.dense_out(x)
      return x

class DropConnect(keras.layers.Dense):
  def __init__(self, *args, **kwargs):
    self.prob = min(1., max(0., kwargs.pop('prob', 0.5)))
    super(DropConnect, self).__init__(*args, **kwargs)

  def call(self, input):
    w = K.dropout(self.kernel, self.prob)
    b = K.dropout(self.bias, self.prob)

    # Same as original
    output = K.dot(input, w)
    if self.use_bias:
      output = K.bias_add(output, b, data_format='channels_last')
    return self.activation(output)


# class printCallback(keras.callbacks.Callback):
#     def on_batch_end(self, batch, logs):
#         weights = model.get_weights()
#         print(weights[1])
#         print(weights[1].shape)
#         exit()