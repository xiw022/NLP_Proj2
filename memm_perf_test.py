import pickle
import data_reader
import memm_prob

import time
import datetime
import json

memm = memm_prob.MEMMProb(None)
reader = data_reader.DataReader()
reader.read_file()

result = {}

for i in range(10):
  t0 = time.time()
  timestamp = int(datetime.datetime.now().strftime('%m%d%H%M%S'))
  print(f"{i}, timestamp={timestamp}")
  ret = reader.split_train_valid(seed=timestamp)
  train = ret["train"]
  memm.train(train, max_iter=10)
  valid = ret["valid"]
  perf = memm.valid_performance(valid)
  result[timestamp] = perf
  print(f"perf={perf}, time for this loop={time.time() - t0}")

with open("memm_performance.json", "w") as fout:
  json.dump(result, fout)