import wfdb
import csv
import pandas as pd
import math
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import random


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

    self.prepare_batch(0.4)


  def prepare_batch(self, irregularity_split=0.5):
    print("Preparing batch...")

    full_i_x_batch = []
    full_i_y_batch = []
    full_j_x_batch = []
    full_j_y_batch = []


    # Individual beats batch
    assert len(self.i_rhythm_batch_values) == len(self.i_rhythm_batch_labels)
    for b_rhyth, l_rhyth in zip(self.i_rhythm_batch_values, self.i_rhythm_batch_labels):
      b_list = list(b_rhyth[:,:,1])
      l_list = list(l_rhyth)

      for i, l in enumerate(l_list): l_list[i] = (0 if l == 'N' else 1)
      for i, b in enumerate(b_list): b_list[i] = b.reshape((b.shape[0], 1))

      full_i_x_batch.extend(b_list)
      full_i_y_batch.extend(l_list)

    assert len(self.j_rhythm_batch_values) == len(self.j_rhythm_batch_labels)
    for b_rhyth, l_rhyth in zip(self.j_rhythm_batch_values, self.j_rhythm_batch_labels):
      b_list = list(b_rhyth[:,:,1])
      l_list = list(l_rhyth)
      
      for i, l in enumerate(l_list): l_list[i] = (0 if l == 'N' else 1)
      for i, b in enumerate(b_list): b_list[i] = b.reshape((b.shape[0], 1))

        
      full_j_x_batch.extend(b_list)
      full_j_y_batch.extend(l_list)

    # irregularity split goes here


    i_only_irreg = [(elem,1) for i, elem in enumerate(full_i_x_batch) if (full_i_y_batch[i])]
    i_only_normal = [(elem,0) for i, elem in enumerate(full_i_x_batch) if (not full_i_y_batch[i])]
    
    j_only_irreg = [(elem,1) for i, elem in enumerate(full_j_x_batch) if (full_j_y_batch[i])]
    j_only_normal = [(elem,0) for i, elem in enumerate(full_j_x_batch) if (not full_j_y_batch[i])]

    i_k_normals = int(len(i_only_irreg)/irregularity_split) - len(i_only_irreg)
    i_normals = random.choices(i_only_normal, k=i_k_normals)

    j_k_normals = int(len(j_only_irreg)/irregularity_split) - len(j_only_irreg)
    j_normals = random.choices(j_only_normal, k=j_k_normals)

    # join normals and irregulars
    i_combined = []
    i_combined.extend(i_normals)
    i_combined.extend(i_only_irreg)

    j_combined = []
    j_combined.extend(j_normals)
    j_combined.extend(j_only_irreg)

    random.shuffle(i_combined)
    random.shuffle(j_combined)


    for i in range(0, len(i_combined)):
      self.i_x_batch.append(i_combined[i][0])
      self.i_y_batch.append(i_combined[i][1])

    for i in range(0, len(j_combined)):
      self.j_x_batch.append(j_combined[i][0])
      self.j_y_batch.append(j_combined[i][1])

    self.i_x_batch = np.array(self.i_x_batch).astype('f')
    self.i_y_batch = np.array(self.i_y_batch).astype('i')
    
    self.j_x_batch = np.array(self.j_x_batch).astype('f')
    self.j_y_batch = np.array(self.j_y_batch).astype('i')


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
      stop = len(y)/self.sample_frequency
      x = np.linspace(0, stop, self.joined_num_samples)

      plt.plot(x, y)
      plt.title(f"Label: {self.j_y_batch[i]}")
      plt.show()



  def j_fold_validation_split(self, test_i:int, k=3):
    """
    Returns x_train, y_train, x_test, y_test
    """
    extra = len(self.j_x_batch) % k
    x_batch_trimmed = self.j_x_batch if extra == 0 else self.j_x_batch[:-extra]
    y_batch_trimmed = self.j_y_batch if extra == 0 else self.j_y_batch[:-extra]

    x_batch_trimmed.astype('f')
    y_batch_trimmed.astype('i')

    print(x_batch_trimmed)
    print(len(y_batch_trimmed))

    size_fold = math.floor(len(self.j_x_batch)/k)
    print(f"Fold Size: {size_fold}")

    #x_train, x_test, y_train, y_test = None
    train_assigned = False
    for i in range(0, len(x_batch_trimmed), size_fold):
      index = int(i/size_fold)
      print(i, "  ", index)

      if index == test_i:
        print("Test fold")
        x_test = x_batch_trimmed[i:i+size_fold]
        y_test = y_batch_trimmed[i:i+size_fold]

      else:
        print("Train fold")
        if not train_assigned:
          train_assigned = True
          x_train = x_batch_trimmed[i:i+size_fold]
          y_train = y_batch_trimmed[i:i+size_fold]
        else:
          np.concatenate((x_train, x_batch_trimmed[i:i+size_fold]), axis=0, dtype='f')
          np.concatenate((y_train, y_batch_trimmed[i:i+size_fold]), axis=0, dtype='i')

    return x_train, x_test, y_train, y_test

  def i_fold_validation_split(self, test_i:int, k=3):
      """
      Returns x_train, y_train, x_test, y_test
      """
      extra = len(self.i_x_batch) % k
      x_batch_trimmed = self.i_x_batch if extra == 0 else self.i_x_batch[:-extra]
      y_batch_trimmed = self.i_y_batch if extra == 0 else self.i_y_batch[:-extra]

      x_batch_trimmed.astype('f')
      y_batch_trimmed.astype('i')

      print(x_batch_trimmed)
      print(len(y_batch_trimmed))

      size_fold = math.floor(len(self.j_x_batch)/k)
      print(f"Fold Size: {size_fold}")

      #x_train, x_test, y_train, y_test = None
      train_assigned = False
      for i in range(0, len(x_batch_trimmed), size_fold):
        index = int(i/size_fold)
        print(i, "  ", index)

        if index == test_i:
          print("Test fold")
          x_test = x_batch_trimmed[i:i+size_fold]
          y_test = y_batch_trimmed[i:i+size_fold]

        else:
          print("Train fold")
          if not train_assigned:
            train_assigned = True
            x_train = x_batch_trimmed[i:i+size_fold]
            y_train = y_batch_trimmed[i:i+size_fold]
          else:
            np.concatenate((x_train, x_batch_trimmed[i:i+size_fold]), axis=0, dtype='f')
            np.concatenate((y_train, y_batch_trimmed[i:i+size_fold]), axis=0, dtype='i')

      return x_train, x_test, y_train, y_test



