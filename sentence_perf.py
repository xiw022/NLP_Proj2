import hmm_prob
import memm_prob
import data_reader
import baseline

import zl_viterbi

# import random
memm = memm_prob.MEMMProb()
memm.load_model("nltk_maxent/nltk_maxent_max_iter_10_1021203327_longer_capital.pickle")
hmm = hmm_prob.HMMProb()
# hmm.train(training_set=train)
hmm.load_model("nltk_hmm/nltk_hmm_1022171807_k_0.1_unk_modeevery_1.pickle")
base = baseline.Baseline()

reader = data_reader.DataReader()
reader.read_file()

import time
t0 = time.time()

import pandas as pd

df = pd.DataFrame(columns="hmm-p hmm-r hmm-f hmm-acc memm-p memm-r memm-f memm-acc base-p base-r base-f base-acc".split(" "))

path = "data_ind_split/simple_valid_portion_0.1_seed_1021205640.pickle"
ret = reader.split_train_valid_by_valid_ind(valid_ind_path=path)
# ret = reader.split_train_valid_by_valid_ind(
#   valid_ind_path="data_ind_split/simple_valid_portion_0.1_seed_1021210356.pickle"
# )
#ret = reader.split_train_valid()
train = ret["train"]
valid = ret["valid"]

base.train(train)


for i, sample in enumerate(valid):
  if i % 100 == 0:
    print(i)
  ret_hmm = zl_viterbi.viterbi(hmm, sample=sample)
  ret_memm = zl_viterbi.viterbi(memm, sample=sample, longer=True, capital=True)
  ret_base = []
  for tok in sample[0].split():
    ret_base.append(base.predict(tok))
  # ret = zl_viterbi.viterbi(memm, sample=sample)
  bios = sample[2].split()

  def prfacc(ret, bios):
    total = correct = p_total = p_correct = r_total = r_correct = 0
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
    if p_total == 0 and r_total == 0:
      p = 1
      r = 1
    elif p_total == 0 or r_total == 0:
      p = 0 if p_total == 0 else p_correct / p_total
      r = 0 if r_total == 0 else r_correct / r_total
    else:
      p = p_correct / p_total
      r = r_correct / r_total
    if p == 0 and r == 0:
      f = 0
    elif p == 0 or r == 0:
      f = p / 2 if p != 0 else r / 2
    else:
      f = 2 * p * r / (p + r)
    acc = correct / total
    return p, r, f, acc

  df.loc[i] = prfacc(ret_hmm, bios) + prfacc(ret_memm, bios) + prfacc(ret_base, bios)


df.to_csv("sentence-wise.csv")

df = pd.read_csv("sentence-wise.csv", index_col=0)
d = {}
cnt = {}
for i in range(1400):
  line = df.loc[i]
  for name in line.keys():
    if name not in d:
      d[name] = [0 for _ in range(1400)]
      cnt[name] = 0
    if line[name] != 0 and line[name] != 1:
      d[name][cnt[name]] = line[name]
      cnt[name] += 1

df2 = pd.DataFrame(columns=list(df))
for i in range(1400):
  df2.loc[i] = [d[col][i] for col in d.keys()]

df2.to_csv("sentence-wise-cleaned.csv")



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