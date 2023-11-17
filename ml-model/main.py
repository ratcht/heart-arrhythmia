import wfdb
import csv
import pandas as pd
import math 


"""
Symbols:
N - Normal
+ - Rhythm Annotation added???


"""



def process_record(patient_number: int):
  patient_record = wfdb.rdrecord(f"data/{patient_number}")

  # Extract patient info, lead names, and ECG data
  leads = patient_record.sig_name
  p_signal = patient_record.p_signal

  # Get freq
  freq = patient_record.fs
  print(freq)

  # Get number of samples in record
  length_record = len(p_signal)

  # Convert p_signal to pandas df
  p_signal_df = pd.DataFrame(p_signal, columns = [*leads])
  p_signal_df.index.name = "sample"

  # Annotation data
  patient_record = wfdb.rdrecord(f"data/{patient_number}")
  annotation = wfdb.rdann(f"data/{patient_number}", "atr")
  annotation_df = pd.DataFrame({"annotation": annotation.symbol}, index=annotation.sample)
  annotation_df.index.name = "sample"

  print(annotation_df['annotation'].unique())

  ecg_data_df = p_signal_df.join(annotation_df)

  # Split into rhythms (where '+' is)

  split_indices = annotation_df.loc[annotation_df["annotation"] == '+'].index.to_list()
  split_indices.insert(0, -1) # Start at -1 to avoid including + annotation
  split_indices.append(length_record)

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
    

  print(batch)



  """
  # Create CSV
  filename = f"{patient_number}.csv"
  outfile = open(filename, "w", newline='')
  out_csv = csv.writer(outfile)

  # Write CSV header with lead names
  out_csv.writerow(leads)

  # Write ECG data to CSV
  """



create_csv(100)

