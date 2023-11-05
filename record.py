import wfdb
import csv
import pandas as pd
import math
import numpy as np


def get_label_table() -> pd.DataFrame:
  return wfdb.io.annotation.ann_label_table

def label_table_to_csv():
  label_table : pd.DataFrame = wfdb.io.annotation.ann_label_table
  label_table.to_csv("label_table.csv")

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


  Methods
  -------

  """
  def __init__(self):
    pass


"""
ECG Record Data
"""
class ECGRecord:
  def __init__(self, patient_number):
    # Load files
    self.patient_record = wfdb.rdrecord(f"data/{patient_number}")
    self.annotation_record = wfdb.rdann(f"data/{patient_number}", "atr")

    # Set Basic Dataframes
    self.set_p_signal_df(self.patient_record)
    self.set_annotation_df(self.annotation_record)

    # Set Properties
    self.sample_frequency: int = self.patient_record.fs
    self.record_length: int = len(self.patient_record.p_signal)

    # Set Rhythm Batch
    self.set_rhythm_batch(self._p_signal_df, self._annotation_df)


  def set_p_signal_df(self, patient_record) :
    leads = patient_record.sig_name
    p_signal = patient_record.p_signal

    # Convert p_signal to pandas df
    self._p_signal_df = pd.DataFrame(p_signal, columns=leads)
    self._p_signal_df.index.name = "sample"


  def set_annotation_df(self, annotation_record) :
    self._annotation_df: pd.DataFrame = pd.DataFrame({"annotation": annotation_record.symbol}, index=annotation_record.sample)
    self._annotation_df.index.name = "sample"
  

  def set_rhythm_batch(self, p_signal_df: pd.DataFrame, annotation_df: pd.DataFrame):
    """
    Split a record into a list containing rhythms

    Returns
    -------
    list[pd.DataFrame]
      A list containing DataFrames, each representing one rhythm split
    """

    ecg_data_df = p_signal_df.join(annotation_df)

    # Split into rhythms (where '+' is)

    split_indices = annotation_df.loc[annotation_df["annotation"] == '+'].index.to_list()
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
      

    self._rhythm_batch: list[pd.DataFrame] = batch
  

  def _split_rhythm(self, rhythm: pd.DataFrame, interval_length:float = 0.8) -> tuple[np.ndarray, np.ndarray]:
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

    annotated_samples: pd.DataFrame = rhythm["annotation"].dropna()


    beats_values = []
    beats_labels = []
    for i, label in annotated_samples.items():
      df = rhythm.loc[i-int(num_samples/2):i+int(num_samples/2)]

      # assert that df only has one annotation in its record
      assert len(df.dropna()) == 1

      row_tuple = list(df.drop("annotation", axis=1).itertuples(index=False, name=None))

      beats_values.append(row_tuple)
      beats_labels.append(label)

    # drop first and last beat to ensure that all beats are same length samples
    del beats_values[0]
    del beats_values[-1]
    del beats_labels[0]
    del beats_labels[-1]

    beats_values = np.array(beats_values, dtype = object)
    # convert sample index to int
    beats_values[:,:,0].astype('int')

    beats_labels = np.array(beats_labels)

    return beats_values, beats_labels


  def _split_rhythm_batch(self) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a rhythm batch into a list of beats

    Returns
    -------
    [[ [ (int, int, int) ] ]], [[ str ]]
    """
    rhythm_batch = self._rhythm_batch

    batch_values = []
    batch_labels = []
    for rhythm in rhythm_batch:
      beats_values, beats_labels = self._split_rhythm(rhythm)
      batch_values.append(beats_values)
      batch_labels.append(beats_labels)
    
    batch_values = np.array(batch_values)
    batch_labels = np.array(batch_labels)


    return batch_values, batch_labels

  """
  Shape of Rhythm Batch:
  (rhythm, beat, samples_in_beat, values)

  Values represents...
  (sample_index, upper_lead, lower_lead)
  """
  
  def to_ecg_data(self) -> ECGData:
    rhythm_batch_values, rhythm_batch_labels = self._split_rhythm_batch()

    

    

ecg_data = ECGRecord(100)
ecg_data.to_ecg_data()


