import hmm_prob
import memm_prob
import data_reader

import zl_viterbi

import gen_kaggle_result

# import random
reader = data_reader.DataReader()
reader.read_file()
ret = reader.split_train_valid_by_valid_ind(
  valid_ind_path="data_ind_split/simple_valid_portion_0.1_seed_1021210356.pickle"
)
#ret = reader.split_train_valid()
train = ret["train"]

# hmm = hmm_prob.HMMProb()
# hmm.train(training_set=train)

memm = memm_prob.MEMMProb()
memm.load_model("nltk_maxent/nltk_maxent_max_iter_10_1021203327_longer_capital.pickle")

reader.read_file("test.txt")
test = reader.data

total = sum([len(sample[0].split()) for sample in test])

result = ["O" for _ in range(total)]

for i, sample in enumerate(test):
  if i % 100 == 0:
    print(i)

  # ret = zl_viterbi.viterbi(hmm, sample=sample)
  ret = zl_viterbi.viterbi(memm, sample=sample, longer=True, capital=True)
  # ret = zl_viterbi.viterbi(memm, sample=sample)
  inds = sample[2].split()

  for pred, ind in zip(ret, inds):
    ind = int(ind)
    result[ind] = pred

ret = gen_kaggle_result.gen_kaggle_result(result, "memm_kaggle.txt")