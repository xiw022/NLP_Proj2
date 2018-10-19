import os
import random

class DataReader:

  def __init__(self):
    self.filepath = "train.txt"
    self.data = None

  def read_file(self, path="train.txt", encoding='utf-8'):
    if path is None:
      path = self.filepath
    if path is None or not os.path.exists(path):
      raise RuntimeError(f"DataReader: cannot open file: \"{path}\"")
    with open(path, encoding=encoding) as fin:
      t = fin.read().strip().split('\n')
    self.data = [[t[i+0], t[i+1], t[i+2]] for i in range(0, len(t), 3)]

  def split_train_valid(self, portion=0.1, seed=None):
    random.seed(seed)
    valid_indices = random.sample(range(len(self.data)), int(portion*len(self.data)))
    valid_indices = set(valid_indices)
    train_indices = set(range(len(self.data))) - valid_indices
    train = [self.data[i] for i in train_indices]
    valid = [self.data[i] for i in valid_indices]
    return {"train": train, "valid": valid}

  def k_fold(self, k=10, order="fixed", seed=None):
    '''
    :param k: number of fold
    :param order: "fixed" for not shuffling, other string for random
    :param seed: random seed
    :return: k-time generator if {"train": train, "valid": valid}
    '''
    all_indices = range(len(self.data))
    if order != "fixed":
      random.seed(seed)
      random.shuffle(all_indices)
    folds = [all_indices[i::k] for i in range(k)]
    all_indices = set(all_indices)
    for i in range(k):
      valid_indices = set(folds[i])
      train_indices = all_indices - valid_indices
      train = [self.data[i] for i in train_indices]
      valid = [self.data[i] for i in valid_indices]
      yield {"train": train, "valid": valid}


# === usage ===
if __name__ == "__main__":
  # do not forget to import
  reader = DataReader()
  #reader.read_file("testfile_for_reader.txt")
  reader.read_file("train.txt")

  import time

  t0 = time.time()
  printed = False
  ret = reader.split_train_valid(portion=0.2)
  print("simple divide")
  #for i, train in enumerate(ret["train"]):
  #  print(f"{i}-th training sample is: {train}")
  #for i, valid in enumerate(ret["valid"]):
  #  print(f"{i}-th validation sample is: {valid}")
  if not printed:
    one_train_sentence = ret["train"][0]
    tokens = one_train_sentence[0]
    pos = one_train_sentence[1]
    ner = one_train_sentence[2]
    print("first training sample:")
    print(f"tokens: \"{tokens}\"", f"pos: \"{pos}\"", f"ner: \"{ner}\"")
    one_valid_sentence = ret["valid"][0]
    tokens = one_valid_sentence[0]
    pos = one_valid_sentence[1]
    ner = one_valid_sentence[2]
    print("first validation sample:")
    print(f"tokens: \"{tokens}\"", f"pos: \"{pos}\"", f"ner: \"{ner}\"")
    printed = True
  simple_split_time = time.time() - t0

  t0 = time.time()
  printed = False
  for j, ret in enumerate(reader.k_fold(k=5)):
    print(f"=== {j}-th fold ===")
    #for i, train in enumerate(ret["train"]):
    #  print(f"{i}-th training sample is: {train}")
    #for i, valid in enumerate(ret["valid"]):
    #  print(f"{i}-th validation sample is: {valid}")
    #print(ret["train"], ret["valid"])
    if not printed:
      one_train_sentence = ret["train"][0]
      tokens = one_train_sentence[0]
      pos = one_train_sentence[1]
      ner = one_train_sentence[2]
      print("first training sample:")
      print(f"tokens: \"{tokens}\"", f"pos: \"{pos}\"", f"ner: \"{ner}\"")
      one_valid_sentence = ret["valid"][0]
      tokens = one_valid_sentence [0]
      pos = one_valid_sentence [1]
      ner = one_valid_sentence [2]
      print("first validation sample:")
      print(f"tokens: \"{tokens}\"", f"pos: \"{pos}\"", f"ner: \"{ner}\"")
      printed = True
  k_fold_time = time.time() - t0

  print(f"simple split: {simple_split_time}, k_fold: {k_fold_time}")


# generate testfile_for_reader.txt
# if __name__ == "__main__":
#   path = "testfile_for_reader.txt"
#   with open(path, "w", encoding="utf-8") as fout:
#     for i in range(100, 300, 10):
#       for j in range(1, 4, 1):
#         fout.write(f"{i+j}\n")
