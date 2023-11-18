import numpy as np
import tensorflow as tf
import keras
from keras import layers
from record import ECGRecord
from ecgdata import ECGData

class Model:
  def __init__(self, input_shape):
    self.model = self.__create_cnn(input_shape, 2)
    
  def __create_cnn(self, input_shape, num_classes) -> keras.models.Model:
    # Create a CNN
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)
  

record = ECGRecord(105)
ecg_data: ECGData = record.to_ecg_data()
#ecg_data.show(0, only_irregularities=True)
#ecg_data.prepare_batch()
#print(ecg_data.x_batch)
ecg_data.show_grouped_beat(0,only_irregularities=False)

#test = Model()