import pickle
import data_reader
import memm_prob

import time
import datetime
import json

import os

def test():

  memm = memm_prob.MEMMProb(None)
  reader = data_reader.DataReader()
  reader.read_file()

  WRITE_FILE = True

  if os.path.isfile("memm_performance_result/memm_performance_all.json"):
    with open("memm_performance_result/memm_performance_all.json", "r") as fin:
      result = json.load(fin)
  else:
    result = {}

  def test(rep=10, longer=False, length=False, pos=False, capital=False):
    global result
    filename = f"memm_performance_local" \
               f"{'_longer' if longer else ''}" \
               f"{'_pos' if pos else ''}" \
               f"{'_length' if length else ''}" \
               f"{'_capital' if capital else ''}.json"
    filename = os.path.join("memm_performance_result", filename)
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
          json.dump(result_local, fout)


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

  # # === tok_i, tag_i_1, tok_i_1, length=True ===
  # print("=== tok_i, tag_i_1, length=True ===")
  # test(rep=10, longer=True, length=True)

  # # === tok_i, tag_i_1, tok_i_1, capital=True ===
  # print("=== tok_i, tag_i_1, capital=True ===")
  # test(rep=10, longer=True, capital=True)

  # # === tok_i, tag_i_1, pos_i, tok_i_1, pos_i_1 (pos=True) ===
  # print("=== tok_i, tag_i_1, pos_i, (pos=True) ===")
  # test(rep=10 , longer=True, pos=True)

  if WRITE_FILE:
    with open("memm_performance_all.json", "w") as fout:
      json.dump(result, fout)


def report():
  with open("memm_performance_result/memm_performance_all.json") as fin:
    j = json.load(fin)

  report = {}
  for timestamp, result in j.items():
    key = "-".join(result["features"])
    if key not in report:
      report[key] = {
        "aggregate": {"p": 0, "r": 0, "f": 0, "acc": 0},
        "raw":{}
      }
    report[key]["raw"][timestamp] = result

  for key, data in report.items():
    raw_cnt = len(data["raw"])
    for _, result in data["raw"].items():
      data["aggregate"]["p"] += result["p"]
      data["aggregate"]["r"] += result["r"]
      data["aggregate"]["f"] += result["f"]
      data["aggregate"]["acc"] += result["acc"]
    data["aggregate"]["p"] /= raw_cnt
    data["aggregate"]["r"] /= raw_cnt
    data["aggregate"]["f"] /= raw_cnt
    data["aggregate"]["acc"] /= raw_cnt

  with open("memm_performance_result/memm_performance_report.json", "w") as fout:
    json.dump(report, fout)


def print_report_file():
  with open("memm_performance_result/memm_performance_report.json") as fin:
    j = json.load(fin)

  for feature, data in j.items():
    print(f"extra features: {feature:>14s}, "
          f"p={data['aggregate']['p']:.6f}, "
          f"r={data['aggregate']['r']:.6f}, "
          f"f={data['aggregate']['f']:.6f}, "
          f"acc={data['aggregate']['acc']:.6f}, ")


def print_raw_data_file():
  import pprint
  with open("memm_performance_result/memm_performance_all.json") as fin:
    j = json.load(fin)
    pprint.pprint(j)
  best = max(((timestamp, result) for timestamp, result in j.items()), key=lambda t: t[1]["f"])
  print(best)


if __name__ == "__main__":
  # test()
  # report()
  # print_report_file()
  print_raw_data_file()