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
  def __init__(self, indiv_rhythm_batch_values, indiv_rhythm_batch_labels, joined_rhythm_batch_values, joined_rhythm_batch_labels, beat_num_samples, joined_num_samples, num_joined, sample_frequency):

    # Base Lists. Contains a batch of rhythms, where each contain a list of beats
    self.i_rhythm_batch_values = indiv_rhythm_batch_values
    self.i_rhythm_batch_labels = indiv_rhythm_batch_labels
    self.j_rhythm_batch_values = joined_rhythm_batch_values
    self.j_rhythm_batch_labels = joined_rhythm_batch_labels

    # Properties
    self.beat_num_samples = beat_num_samples
    self.sample_frequency = sample_frequency
    self.joined_num_samples = joined_num_samples
    self.num_joined = num_joined

    # Batches
    self.i_x_batch = []
    self.i_y_batch = []
    self.j_x_batch = []
    self.j_y_batch = []


  def prepare_batch(self):
    print("Preparing batch...")

    # Individual beats batch
    assert len(self.i_rhythm_batch_values) == len(self.i_rhythm_batch_labels)
    for b_rhyth, l_rhyth in zip(self.i_rhythm_batch_values, self.i_rhythm_batch_labels):
      b_list = list(b_rhyth[:,:,1])
      l_list = list(l_rhyth)

      for i, l in enumerate(l_list): l_list[i] = 0 if l == 'N' else 1

      self.i_x_batch.extend(b_list)
      self.i_y_batch.extend(l_list)

    assert len(self.j_rhythm_batch_values) == len(self.j_rhythm_batch_labels)
    for b_rhyth, l_rhyth in zip(self.j_rhythm_batch_values, self.j_rhythm_batch_labels):
      b_list = list(b_rhyth[:,:,1])
      l_list = list(l_rhyth)
      
      for i, l in enumerate(l_list): l_list[i] = 0 if l == 'N' else 1
        
      self.j_x_batch.extend(b_list)
      self.j_y_batch.extend(l_list)

  def show_individual_beat(self, only_irregularities=False):
    print("\n\n\nTEST")
    for i in range(0, len(self.i_y_batch)):
      y = self.i_x_batch[i]

      if only_irregularities and self.i_y_batch[i] == "N": continue
      print("\n\nLABEL:")
      print(self.i_y_batch[i])
      stop = len(y)/self.sample_frequency
      x = np.linspace(0, stop, self.beat_num_samples)

      plt.plot(x, y)
      plt.title(f"Label: {self.i_y_batch[i]}")
      plt.show()

  def show_grouped_beat(self, only_irregularities=False):
    print("\n\n\nTEST")
    for i in range(0, len(self.j_y_batch)):
      y = self.j_x_batch[i]

      if only_irregularities and self.j_y_batch[i] == "N": continue
      print("\n\nLABEL:")
      print(self.j_y_batch[i])
      stop = len(y)/self.sample_frequency
      x = np.linspace(0, stop, self.joined_num_samples)

      plt.plot(x, y)
      plt.title(f"Label: {self.j_y_batch[i]}")
      plt.show()

