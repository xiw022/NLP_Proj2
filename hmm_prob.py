import bigram
import unigram
import pickle
import os
import datetime
from collections import Counter
from itertools import chain

class HMMProb:
  def __init__(self, k=0.1, unk_mode="every_1"):
    self.k = k
    self.unk_mode = unk_mode
    self.bigram = None
    self.tag2tok = {}

  def train(self, training_set,
            timestamp=int(datetime.datetime.now().strftime('%m%d%H%M%S')),
            k=None, unk_mode=None):
    self.tag2tok = {}

    k = self.k if k is None else k
    unk_mode = self.unk_mode if unk_mode is None else unk_mode

    gen = []

    for i, sample in enumerate(training_set):
      toks, _, bios = sample
      bios_list = bios.split()

      # store toks for each tag, for P(w | t)
      for tok, bio in zip(toks.split(), bios_list):
        if bio not in self.tag2tok:
          self.tag2tok[bio] = []
        self.tag2tok[bio].append(tok)

      # chain for bigram model
      gen = chain(gen, ["O"], bios_list)

    # train bigram
    # use unk_mode=none because tag is a closed set, nothing out of vocab
    self.bigram = bigram.BigramModel(tokens=gen, k=k, unk_mode="none")

    # train for P(w | t) using unigram
    for tag, toks in self.tag2tok.items():
      self.tag2tok[tag] = unigram.UnigramModel(toks, k=k, unk_mode=unk_mode)

    if not os.path.exists("nltk_hmm"):
      os.makedirs("nltk_hmm")
    filepath = os.path.join("nltk_hmm",
                            f"nltk_hmm_{timestamp}"
                            f"_k_{k}_unk_mode{unk_mode}.pickle")
    model = {
      "bigram" : self.bigram,
      "tag2tok": self.tag2tok,
      "k"      : self.k,
      "unk_mode":self.unk_mode,
    }
    with open(filepath, "wb") as fout:
      pickle.dump(model, fout)
      print("dump model to file:", filepath)


  def load_model(self, model_path):
    with open(model_path, "rb") as fin:
      model = pickle.load(fin)
    self.bigram = model["bigram"]
    self.tag2tok = model["tag2tok"]
    self.k = model["k"]
    self.unk_mode = model["unk_mode"]

  def calc_prob(self, tok_i, bio_i, bio_i_1, k=None):
    k = self.k if k is None else k
    # trans: P(t_i | t_i-1)
    trans = self.bigram.calculate_prob(prev=bio_i_1, curr=bio_i, k=k)
    # emit:  P(w_i | t_i)
    emit = self.tag2tok[bio_i].calculate_prob(tok_i, k=k)
    return trans * emit

  def calc_probs(self, tok_i, bio_i_1, k=None):
    k = self.k if k is None else k
    tags = self.tag2tok.keys()
    return {
      bio_i : self.bigram.calculate_prob(prev=bio_i_1, curr=bio_i, k=k) \
              * self.tag2tok[bio_i].calculate_prob(tok_i, k=k)
      for bio_i in tags
    }


  def predict(self, tok_i, bio_i_1, k=None):
    k = self.k if k is None else k
    probs = self.calc_probs(tok_i, bio_i_1, k=k)
    tag = ""
    prob = 0
    for k, v in probs.items():
      if v > prob:
        prob = v
        tag = k
    return tag


  def valid_performance(self, validation_set, k=None):
    k = self.k if k is None else k
    print("performance upon validation with"
          f"_k_{k}_unk_mode{self.unk_mode}")

    total = 0
    correct = 0
    p_total = 0
    p_correct = 0
    r_total = 0
    r_correct = 0
    for i, sample in enumerate(validation_set):
      #if i % 100 == 0:
      #  print(i)
      toks, poss, bios = sample
      last_bio = "O"
      #print(toks)
      for item in zip(toks.split(), bios.split(), poss.split()):
        # item[0] is tok, item[1] is bio, item[2] is pos
        pred = self.predict(item[0], last_bio, k=k)
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
        print(pred, item[1], end="; ")
        last_bio = item[1]
      #print(f"=== {i:>4d}: {correct/total} === \n")
    p = p_correct/p_total
    r = r_correct/r_total
    return {
      "p"  : p,
      "r"  : r,
      "f"  : 2*p*r/(p+r),
      "acc": correct/total,
    }

if __name__ == "__main__":
  import time
  import data_reader
  timestamp = int(datetime.datetime.now().strftime('%m%d%H%M%S'))
  t0 = time.time()
  print(datetime.datetime.now().strftime("starts at %H:%M:%S"))
  print("=== reading training set ===")
  reader = data_reader.DataReader()
  reader.read_file()
  ret = reader.split_train_valid(seed=timestamp)
  train = ret["train"]
  valid = ret["valid"]

  print(f"=== time used {time.time() - t0}")
  print("=== training hmm_prob ===")
  hmm = HMMProb()
  hmm.train(train)
  print(f"=== time used {time.time() - t0}")
  ret = hmm.valid_performance(valid)
  print(ret)
  print("done")
