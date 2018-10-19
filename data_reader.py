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
    train = (self.data[i] for i in train_indices)
    valid = (self.data[i] for i in valid_indices)
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
      train = (self.data[i] for i in train_indices)
      valid = (self.data[i] for i in valid_indices)
      yield {"train": train, "valid": valid}


# === usage ===
if __name__ == "__main__":
  # do not forget to import
  reader = DataReader()
  reader.read_file("testfile_for_reader.txt")
  ret = reader.split_train_valid(portion=0.2)
  print("simple divide")
  for i, train in enumerate(ret["train"]):
    print(f"{i}-th training sample is: {train}")
  for i, valid in enumerate(ret["valid"]):
    print(f"{i}-th validation sample is: {valid}")

  for j, ret in enumerate(reader.k_fold(k=5)):
    print(f"=== {j}-th fold ===")
    for i, train in enumerate(ret["train"]):
      print(f"{i}-th training sample is: {train}")
    for i, valid in enumerate(ret["valid"]):
      print(f"{i}-th validation sample is: {valid}")

# generate testfile_for_reader.txt
# if __name__ == "__main__":
#   path = "testfile_for_reader.txt"
#   with open(path, "w", encoding="utf-8") as fout:
#     for i in range(100, 300, 10):
#       for j in range(1, 4, 1):
#         fout.write(f"{i+j}\n")
