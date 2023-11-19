import wfdb
import csv
import pandas as pd
import math
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import math
from ecgdata import ECGData


def get_label_table() -> pd.DataFrame:
  return wfdb.io.annotation.ann_label_table

def label_table_to_csv():
  label_table: pd.DataFrame = get_label_table()
  label_table.to_csv("label_table.csv")


"""
ECG Record Data
"""
class ECGRecord:
  """
  A class to process an ECG record given a patient record

  ...

  Attributes
  -------

  Methods
  -------

  """
  def __init__(self, patient_number):
    # Load files
    self.patient_record = wfdb.rdrecord(f"data/{patient_number}")
    self.annotation_record = wfdb.rdann(f"data/{patient_number}", "atr")

    # Set Properties
    self.sample_frequency: int = self.patient_record.fs
    self.leads: list = self.patient_record.sig_name
    self.record_length: int = len(self.patient_record.p_signal)

    # Set Basic Dataframes
    self.set_p_signal_df(self.patient_record)
    self.set_annotation_df(self.annotation_record)

    # Set Rhythm Batch
    self.set_rhythm_batch(self.__p_signal_df, self.__annotation_df)


  def set_p_signal_df(self, patient_record) :
    p_signal = patient_record.p_signal

    # Convert p_signal to pandas df
    self.__p_signal_df = pd.DataFrame(p_signal, columns=self.leads)
    self.__p_signal_df.index.name = "sample"


  def set_annotation_df(self, annotation_record) :
    self.__annotation_df: pd.DataFrame = pd.DataFrame({"annotation": annotation_record.symbol}, index=annotation_record.sample)
    self.__annotation_df.index.name = "sample"
  

  def set_rhythm_batch(self, p_signal_df: pd.DataFrame, annotation_df: pd.DataFrame):
    """
    Split a record into a list containing rhythms

    Returns
    -------
    list[pd.DataFrame]
      A list containing DataFrames, each representing one rhythm split
    """

    ecg_data_df = p_signal_df.join(annotation_df)

    # Split into rhythms (where '+' or '~' is)

    split_indices = annotation_df.loc[annotation_df["annotation"].isin(['+','~'])].index.to_list()
    split_indices.insert(0, -1) # Start at -1 to avoid including + annotation
    split_indices.append(self.record_length)

    batch = []

    for i in range(0,len(split_indices)-1):
      # -1 and +1 to start and end in order to skip the row containing '+'
      start_index = split_indices[i]+1
      end_index = split_indices[i+1]-1

      df = ecg_data_df.loc[start_index:end_index]

      # Check if the sample contains any annotations, i.e, doesn't only contain NaN annotations
      unique_labels = df['annotation'].unique()

      # check conditions:
      # 1. if all values are floats (since nan is a float type)
      # 2. if all values are nan
      # 3. if length is longer than 0
      if not(all(isinstance(l, float) for l in unique_labels)
            and all(math.isnan(l) for l in unique_labels)
            and len(unique_labels) > 0):
        batch.append(df.reset_index()) # Reset index to start at 0
      

    self.__rhythm_batch: list[pd.DataFrame] = batch
  

  def __split_rhythm(self, rhythm: pd.DataFrame, normalize, interval_length:float = 1, num_beats_joined = 3, num_samples_joined=815) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a rhythm into nested lists, each representing a beat centered around the annotation sample

    Parameters
    -------
    interval_length: float
      Length of interval centered around annotation. In seconds.
    
    Returns
    -------
    [ [ (int, int, int) ] ], [ str ]
      Two lists for each beat. One contains the values in tuples of (sample_index, upper_lead, lower_lead). The second contains the corresponding label to each beat.
    """
    num_samples: int = int(interval_length*self.sample_frequency)

    annotated_samples: pd.Series = rhythm["annotation"].dropna()

    indiv_beats_values = []
    indiv_beats_labels = []
    joined_beats_values = []
    joined_beats_labels = []
    skip_this=False
    for i, label in annotated_samples.items():
      if skip_this: 
        skip_this = False
        continue
      indiv_df = rhythm.loc[i-math.floor(num_samples/2):i+math.ceil(num_samples/2)-1]

      # assert that df only has one annotation in its record
      if not (len(indiv_df.dropna()) == 1):
        label = indiv_df["annotation"].dropna().loc[indiv_df["annotation"] != "N"].values[0]
        skip_this = True

      row_tuple = list(indiv_df.drop("annotation", axis=1).itertuples(index=False, name=None))
      indiv_beats_values.append(row_tuple)
      indiv_beats_labels.append(label)

    # drop first and last beat to ensure that all beats are same length samples
    del indiv_beats_values[0]
    del indiv_beats_values[-1]
    del indiv_beats_labels[0]
    del indiv_beats_labels[-1]

    # Check if all beats have same num samples
    comparator_length = len(indiv_beats_values[0])
    if not all(len(beat) == comparator_length for beat in indiv_beats_values): raise Exception("Not all values have same sample length")
    self.beat_num_samples = comparator_length

    indiv_beats_values = np.array(indiv_beats_values, dtype = object)
    # convert sample index to int
    indiv_beats_values[:,:,0].astype('int')

    for i in range(0, len(indiv_beats_values)-(num_beats_joined-1), num_beats_joined):
      start_sample = indiv_beats_values[i][0][0]
      end_sample = indiv_beats_values[i+(num_beats_joined-1)][len(indiv_beats_values[i+(num_beats_joined-1)])-1][0]

      diff_samples = end_sample-start_sample
      error = diff_samples - num_samples_joined + 1
      #assert error > 0, f"Diff samples: {diff_samples}, Num samples: {num_samples_joined}"

      joined_df: pd.DataFrame = rhythm.loc[(rhythm["sample"] >= (start_sample + math.floor(error/2))) & (rhythm["sample"] <= (end_sample - math.ceil(error/2)))]
      
      labels = list(joined_df["annotation"].dropna().unique())
      labels.remove('N')
      label = 'N' if len(labels) == 0 else labels[0]

      row_tuple = list(joined_df.drop("annotation", axis=1).itertuples(index=False, name=None))
      joined_beats_values.append(row_tuple)
      joined_beats_labels.append(label)

    # Check if all beats have same num samples
    comparator_length = len(joined_beats_values[0])
    if not all(len(beat) == comparator_length for beat in joined_beats_values): raise Exception("Not all values have same sample length")
    self.joined_beat_num_samples = comparator_length

    joined_beats_values = np.array(joined_beats_values, dtype = object)
    # convert sample index to int
    joined_beats_values[:,:,0].astype('int')


    # Normalize Data
    # Add the abs of the lowest negative value to each entry such that all values are now positive
    if normalize:
      scaler = preprocessing.MinMaxScaler()
      indiv_beats_values[:,:,1] += abs(np.min(indiv_beats_values[:,:,1]))
      indiv_beats_values[:,:,1] = scaler.fit_transform(indiv_beats_values[:,:,1])

      indiv_beats_values[:,:,2] += abs(np.min(indiv_beats_values[:,:,2]))
      indiv_beats_values[:,:,2] = scaler.fit_transform(indiv_beats_values[:,:,2])

      joined_beats_values[:,:,1] += abs(np.min(joined_beats_values[:,:,1]))
      joined_beats_values[:,:,1] = scaler.fit_transform(joined_beats_values[:,:,1])

      joined_beats_values[:,:,2] += abs(np.min(joined_beats_values[:,:,2]))
      joined_beats_values[:,:,2] = scaler.fit_transform(joined_beats_values[:,:,2])

    indiv_beats_labels = np.array(indiv_beats_labels)
    joined_beats_labels = np.array(joined_beats_labels)
    return indiv_beats_values, indiv_beats_labels, joined_beats_values, joined_beats_labels


  def __split_rhythm_batch(self, join, joined_samples, normalize=False) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Split a rhythm batch into a list of beats

    Returns
    -------
    [[ [ (int, int, int) ] ]], [[ str ]]
    """
    rhythm_batch = self.__rhythm_batch

    individual_batch_values = []
    individual_batch_labels = []

    joined_batch_values = []
    joined_batch_labels = []

    for rhythm in rhythm_batch:
      if len(rhythm) < (360*join): continue
      indiv_beats_values, indiv_beats_labels, joined_beats_values, joined_beats_labels = self.__split_rhythm(rhythm, normalize=normalize, interval_length=1, num_beats_joined=join, num_samples_joined=joined_samples)
      individual_batch_values.append(indiv_beats_values)
      individual_batch_labels.append(indiv_beats_labels)
      joined_batch_values.append(joined_beats_values)
      joined_batch_labels.append(joined_beats_labels)

  
    return individual_batch_values, individual_batch_labels, joined_batch_values, joined_batch_labels

  """
  Shape of Rhythm Batch:
  (rhythm, beat, samples_in_beat, values)
  (x, y, 289, 3)

  
  Values represents...
  (sample_index, upper_lead, lower_lead)
  """
  
  # 3 beats -> 720
  # 5 beats -> 1350

  def to_ecg_data(self, normalize=True) -> ECGData:
    rhythm_indiv_batch_values, rhythm_indiv_batch_labels, rhythm_joined_batch_values, rhythm_joined_batch_labels = self.__split_rhythm_batch(normalize=normalize, join=5, joined_samples=1350)
    return ECGData(rhythm_indiv_batch_values, rhythm_indiv_batch_labels,
                   rhythm_joined_batch_values, rhythm_joined_batch_labels, 
                   self.beat_num_samples, 1350, 5 ,self.sample_frequency)

