import wfdb
import csv
import pandas as pd


def create_csv(patient_number: int):
  patient_record = wfdb.rdrecord(f"data/{100}")

  # Extract patient info, lead names, and ECG data
  patient_number = patient_record.record_name
  leads = patient_record.sig_name
  ecg_data = patient_record.p_signal

  # Get freq
  freq = patient_record.fs

  # Create CSV
  filename = f"{patient_number}.csv"
  outfile = open(filename, "w", newline='')
  out_csv = csv.writer(outfile)

  # Write CSV header with lead names
  out_csv.writerow(leads)

  # Write ECG data to CSV
  for row in ecg_data:
    out_csv.writerow(row)

patient_record = wfdb.rdrecord(f"data/{100}")
annotation = wfdb.rdann(f"data/{100}", "atr")
print(annotation.sample)
