import wfdb
import csv
import pandas as pd
import math
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

"""
Object of Neural Network Input Data
"""
class ECGData:
  """
  A class used to represent the ECG input data for the neural network

  ...

  Attributes
  ----------
  beats: list[pd.DataFrame]
    A list of dataframe objects representing the beat

  beat_length: int
    Number of samples in a single beat
  
  annotation_list: list[str]

  x_batch:
  y_batch:

  Classification Notes:
    y will either be 0 or 1. 0 for normal, 1 for arrhythmia

  Methods
  -------

  """
  def __init__(self, rhythm_batch_values, rhythm_batch_labels, beat_num_samples, sample_frequency):

    # Base Lists. Contains a batch of rhythms, where each contain a list of beats
    self.rhythm_batch_values = rhythm_batch_values
    self.rhythm_batch_labels = rhythm_batch_labels

    # Properties
    self.beat_num_samples = beat_num_samples
    self.sample_frequency = sample_frequency

    # BATCH
    self.x_batch = []
    self.y_batch = []

  def join_beats(self, rhythm_values: np.ndarray, rhythm_labels: np.ndarray, num_beats) -> tuple[list[np.ndarray], list[str]]:
    """
    Given a rhythm, join individual beats into a set of x beats. The labels will be joined together, so if all are N, then the label for the group of beats will be N. Otherwise, it will be A
    """
    joined_beats_values = []
    joined_beats_labels = []

    # turn rhythm into list
    rhythm_values: list = rhythm_values.tolist()
    rhythm_labels: list = rhythm_labels.tolist()

    # check if rhythm can be incremented nicely. Drop last values if not.
    if not (len(rhythm_values) % num_beats == 0):
      num_extra = (len(rhythm_values) % num_beats)
      
      del rhythm_values[len(rhythm_values) - num_extra : len(rhythm_values)]
      del rhythm_labels[len(rhythm_labels) - num_extra : len(rhythm_labels)]


    for i in range(0,len(rhythm_values), num_beats):
      joined_array_values = np.array(rhythm_values[i:i+num_beats])

      # Join labels. If no arrhythmia, 0, else 1
      joined_labels = rhythm_labels[i:i+num_beats]

      combined_label = 0

      # If one of beats is not normal
      if not all(item == 'N' for item in joined_labels): combined_label = 1

      joined_beats_values.append(joined_array_values)
      joined_beats_labels.append(combined_label)
  
    print(joined_beats_labels)


        

  def show(self):
    y = self.rhythm_batch_values[:,:,:,1][0][1]

    stop = len(y)/self.sample_frequency
    x = np.linspace(0, stop, 289)

    plt.plot(x, y)
    plt.show()

