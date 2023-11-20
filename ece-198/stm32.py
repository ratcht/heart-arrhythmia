from mlmodel.model import Model, send_make_prediction
from mlmodel.record import ECGData, ECGRecord
import serial
from concurrent.futures.thread import ThreadPoolExecutor
import concurrent.futures
import keras
import time
import tensorflow as tf


def read_port():
  with serial.Serial('/dev/tty.usbmodem1203', 115200, timeout=1) as ser:
    while True:
      line = ser.readline()
      print(line)

def write_port_loop(bytes):
  with serial.Serial('/dev/tty.usbmodem1203', 115200) as ser:
    for line in bytes:
      ser.write(line)
      #time.sleep(1)


def write_port(bytes):
  with serial.Serial('/dev/tty.usbmodem1203', 115200) as ser:
    ser.write(bytes)
    #time.sleep(1)

def feed_stm32(limit_beats, j_x_batch, j_y_batch) -> float:
  tally = 0
  length_batch = len(j_y_batch)

  assert limit_beats <= length_batch
  for i, beat in enumerate(j_x_batch):
    correct_result = j_y_batch[i]
    prediction = send_make_prediction(beat, j_y_batch[i])

    is_correct = (prediction == correct_result)

    if is_correct:
      tally += 1

    print(f"Beat Group {i}:")
    print(f"Model Prediction {prediction}, Correct Result: {correct_result}. Is correct: {is_correct}")
    print("---------------------\n")

    if i > limit_beats:
      break
  

  accuracy = tally/limit_beats
  print("------------------------")
  print(f"Final Accuracy: {accuracy}")


def feed_stm32_test_3(limit_beats, j_x_batch, j_y_batch) -> float:
  tally = 0
  length_batch = len(j_y_batch)

  is_arrhythmia = 0
  start_time = 0

  assert limit_beats <= length_batch
  for i, beat in enumerate(j_x_batch):
    prediction = send_make_prediction(beat, j_y_batch[i])

    if prediction == 1:
      is_arrhythmia = 1
    


    print(f"Beat Group {i}:")
    print(f"Model Prediction {prediction}")

    # every 6 beat groups process a final result
    if i % 10 == 0:
      end_time = time.time()
      if is_arrhythmia:
        print("====Arrhythmia Detected within last 50 beats====")
        print(f"Time Taken for Result: {end_time - start_time}\n\n")
        is_arrhythmia = 0

      else:
        print("====No Arrhythmia Detected within last 50 beats====")
        print(f"Time Taken for Result: {end_time - start_time}\n\n")
        is_arrhythmia = 0
      start_time = time.time()

    if i > limit_beats:
      break
  
