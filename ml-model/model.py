import numpy as np
import tensorflow as tf
import keras
from keras import layers
from record import ECGRecord
import matplotlib.pyplot as plt
from ecgdata import ECGData
import statistics


class Model:
  def __init__(self, input_shape):
    self.model = self.make_cnn(input_shape, 2)
    
  def make_cnn(self, input_shape, num_classes) -> keras.models.Model:
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
  
def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


# root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

# x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
# x_test, y_test = readucr(root_url + "FordA_TEST.tsv")


# x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
# x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))





# print(y_train)
# print(y_train.shape)
# print(np.count_nonzero(ecg_data.j_y_batch))

# print(ecg_data.j_y_batch.shape)



# record = ECGRecord(105)
# ecg_data: ECGData = record.to_ecg_data()
# ecg_data.prepare_batch()
# ecg_data.show_grouped_beat(only_irregularities=False)

record = ECGRecord(100)
ecg_data: ECGData = record.to_ecg_data()

