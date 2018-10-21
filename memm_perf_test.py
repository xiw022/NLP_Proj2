import pickle
import data_reader
import memm_prob

import time
import datetime
import json

import os

memm = memm_prob.MEMMProb(None)
reader = data_reader.DataReader()
reader.read_file()

WRITE_FILE = True

if os.path.isfile("memm_performance_all.json"):
  with open("memm_performance_all.json", "r") as fin:
    result = json.load(fin)
else:
  result = {}

def test(rep=10, longer=False, length=False, pos=False, capital=False):
  global result
  filename = f"memm_performance_local \
               {'_longer' if longer else ''} \
               {'_pos' if pos else ''} \
               {'_length' if length else ''} \
               {'_capital' if capital else ''}.json"
  if os.path.isfile(filename):
    with open(filename, "r") as fin:
      result_local = json.load(fin)
  else:
    result_local = {}

  for i in range(rep):
    t0 = time.time()
    timestamp = int(datetime.datetime.now().strftime('%m%d%H%M%S'))
    print(f"{i}, timestamp={timestamp}", end="; ")
    ret = reader.split_train_valid(seed=timestamp)
    train = ret["train"]
    memm.train(train, max_iter=10, longer=True)
    valid = ret["valid"]
    perf = memm.valid_performance(valid)
    
    result[timestamp] = perf
    result[timestamp]["features"] = []
    if longer:
      result[timestamp]["features"].append("longer")
    if length:
      result[timestamp]["features"].append("length")
    if pos:
      result[timestamp]["features"].append("pos")
    if capital:
      result[timestamp]["features"].append("capital")
    
    result_local[timestamp] = perf
    print(f"perf={perf}, time for this loop={time.time() - t0}")
    if WRITE_FILE:
      with open(filename, "w") as fout:
        json.dump(result, fout)


WRITE_FILE = False

# === tok_i, tag_i_1 ===
print("=== tok_i, tag_i_1 ===")
test(rep=10)

# === tok_i, tag_i_1, tok_i_1 (longer=True) ===
print("=== tok_i, tag_i_1, tok_i_1 (longer=True) ===")
test(rep=10, longer=True)

# === tok_i, tag_i_1, length=True ===
print("=== tok_i, tag_i_1, length=True ===")
test(rep=10, length=True)

# === tok_i, tag_i_1, capital=True ===
print("=== tok_i, tag_i_1, capital=True ===")
test(rep=10, capital=True)

# === tok_i, tag_i_1, pos_i, (pos=True) ===
print("=== tok_i, tag_i_1, pos_i, (pos=True) ===")
test(rep=10, pos=True)

# === tok_i, tag_i_1, tok_i_1, length=True ===
print("=== tok_i, tag_i_1, length=True ===")
test(rep=10, longer=True, length=True)

# === tok_i, tag_i_1, tok_i_1, capital=True ===
print("=== tok_i, tag_i_1, capital=True ===")
test(rep=10, longer=True, capital=True)

# === tok_i, tag_i_1, pos_i, tok_i_1, pos_i_1 (pos=True) ===
print("=== tok_i, tag_i_1, pos_i, (pos=True) ===")
test(rep=10 , longer=True, pos=True)

if WRITE_FILE:
  with open("memm_performance_all.json", "w") as fout:
    json.dump(result, fout)