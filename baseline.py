from collections import Counter
import pandas as pd

class Baseline:
  def __init__(self):
    self.data = {}

  def train(self, training_set):
    # get all tags of each word
    if len(self.data) > 0:
      self.data = {}
    for i, sample in enumerate(training_set):
      toks, poss, bios = sample
      for tok, bio in zip(toks.split(), bios.split()):
        if tok not in self.data:
          self.data[tok] = []
        self.data[tok].append(bio)

    # calculate tag freq of each word, select the max one
    for tok in self.data:
      counter = Counter(self.data[tok])
      choice = max(counter, key=counter.get)
      self.data[tok] = choice

    print("training step finished")

  def predict(self, token):
    return self.data[token] if token in self.data else "O"

  def valid_performance(self, validation_set):
    total = 0
    correct = 0
    p_total = 0
    p_correct = 0
    r_total = 0
    r_correct = 0
    for i, sample in enumerate(validation_set):
      toks, poss, bios = sample
      for item in zip(toks.split(), bios.split()):
        # item[0] is tok, item[1] is bio
        pred = self.predict(item[0])
        # p
        if pred != "O":
          p_total += 1
          if pred == item[1]:
            p_correct += 1
        # r
        if item[1] != "O":
          r_total += 1
          if pred == item[1]:
            r_correct += 1
        # acc
        if pred == item[1]:
          correct += 1
        total += 1
    p = p_correct/p_total
    r = r_correct/r_total
    return {
      "p"  : p,
      "r"  : r,
      "f"  : 2*p*r/(p+r),
      "acc": correct/total,
    }


def performance_test(seed=None, rep=10):
  import data_reader
  #import random
  reader = data_reader.DataReader()
  reader.read_file()

  baseline = Baseline()

  p = 0
  r = 0
  f = 0
  acc = 0

  df = pd.DataFrame(columns="p r f acc".split(" "))

  for i in range(rep):
    ret = reader.split_train_valid(seed=seed)
    train = ret["train"]
    baseline.train(train)
    valid = ret["valid"]
    perf = baseline.valid_performance(valid)
    p += perf["p"]
    r += perf["r"]
    f += perf["f"]
    acc += perf["acc"]
    df.loc[i] = perf["p"], perf["r"], perf["f"], perf["acc"]
  print(p/rep, r/rep, f/rep, acc/rep)
  df.to_csv("baseline_perf_report.csv")

if __name__ == "__main__":
  performance_test()