import hmm_prob
import memm_prob
import data_reader

import zl_viterbi

# import random
memm = memm_prob.MEMMProb()
memm.load_model("nltk_maxent/nltk_maxent_max_iter_10_1022193825_longer_pos_capital.pickle")

reader = data_reader.DataReader()
reader.read_file()

import time
t0 = time.time()

import os

for root, dirs, files in os.walk("data_ind_split/temp"):
  print(len(files))
  for file_cnt, file in enumerate(files):
    if file_cnt > 15:
      break
    print(file[30:40], end=": ")
    path = os.path.join(root, file)
    ret = reader.split_train_valid_by_valid_ind(valid_ind_path=path)
    # ret = reader.split_train_valid_by_valid_ind(
    #   valid_ind_path="data_ind_split/simple_valid_portion_0.1_seed_1021210356.pickle"
    # )
    #ret = reader.split_train_valid()
    train = ret["train"]
    valid = ret["valid"]

    # hmm = hmm_prob.HMMProb()
    # hmm.train(training_set=train)

    total = 0
    correct = 0
    p_total = 0
    p_correct = 0
    r_total = 0
    r_correct = 0

    for i, sample in enumerate(valid):
      # if i % 100 == 0:
      #   print(i)

      # ret = zl_viterbi.viterbi(hmm, sample=sample)
      ret = zl_viterbi.viterbi(memm, sample=sample, longer=True, capital=True, pos=True)
      # ret = zl_viterbi.viterbi(memm, sample=sample)
      bios = sample[2].split()

      for pred, bio in zip(ret, bios):
        if pred != "O":
          p_total += 1
          if pred == bio:
            p_correct += 1
        if bio != "O":
          r_total += 1
          if pred == bio:
            r_correct += 1
        if bio == pred:
          correct += 1
        total += 1

    p = p_correct / p_total
    r = r_correct / r_total
    print(2*p*r / (p+r))
    # print(
    #   {
    #     "p": p,
    #     "r": r,
    #     "f": 2 * p * r / (p + r),
    #     "acc": correct / total,
    #   }
    # )

# print(time.time() - t0)

# sample = None
# for v in valid:
#   if "B" in v[2]:
#     sample = v
#     break
# else:
#   print("ERROR!")
#   exit(-1)
#
# ret = zl_viterbi.viterbi(hmm, sample=sample)
#
# print(sample)
# print(ret)
# print(sample)