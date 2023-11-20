from mlmodel.model import Model
from mlmodel.record import ECGData, ECGRecord
import serial
from concurrent.futures.thread import ThreadPoolExecutor
import concurrent.futures
import keras
import time
import tensorflow as tf
from stm32 import *
import decorators


def send_to_stm():
  model_bytes = tf.load_model("best_model.tflite")
  write_port("b".encode()) # starting char
  for line in model_bytes:
    write_port(line.encode())
  write_port("e".encode()) # ending char


def run_pool():
  executor = ThreadPoolExecutor(max_workers=10)
  futures = [executor.submit(read_port)]
  done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

def prepare_test():
  ecg_record = ECGRecord(105)
  ecg_data: ECGData = ecg_record.to_ecg_data()
  return ecg_data


def test_1():
  # memory test
  # Check stm32 memory when running

  pass

def test_2():
  # > 54% Accuracy on arrhythmia detection
  # show that prepare batch is 5050
  limit_beats = 30
  ecg_data = prepare_test()
  print("ECG Dataset: ")
  print(ecg_data.j_y_batch)

  feed_stm32(limit_beats, ecg_data.j_x_batch, ecg_data.j_y_batch)

decorators.time_to_execute
def test_3():
  # > 54% Accuracy on arrhythmia detection
  # show that prepare batch is 5050
  limit_beats = 30
  ecg_data = prepare_test()
  print("ECG Dataset: ")
  print(ecg_data.j_y_batch)

  feed_stm32_test_3(limit_beats, ecg_data.j_x_batch, ecg_data.j_y_batch)


test_2()