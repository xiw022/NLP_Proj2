import numpy as np
import nltk
from gensim.models import KeyedVectors
from nltk.classify import MaxentClassifier
import pickle
import datetime
import os

import memm_util


class MEMMProb:
  def __init__(self, word2vec, w2v_len=300, is_bin=True):
    self.training_set = None
    if word2vec is None:
      self.w2v = None
    else:
      self.w2v = KeyedVectors.load_word2vec_format(word2vec, binary=is_bin)
    self.w2v_len = w2v_len
    self.maxent = None

  def gen_feature(self, tok, bio_i_1):
    '''
    #try:
      curr_tok = self.w2v.get_vector(tok) \
        if self.w2v is not None and tok in self.w2v.vocab \
        else np.zeros((self.w2v_len,))
      # "curr_bio": memm_util.bio2vec(bio_i),
      prev_bio = memm_util.bio2vec(bio_i_1)

      #print(curr_tok.shape)
      #print(prev_bio.shape)

      ret = np.concatenate((curr_tok, prev_bio), axis=0)

      return {f"{i}" : ret[i] for i in range(ret.shape[0])}

    #except Exception as e:
    #  print(e)
    #  print(f"tok: \"{tok}\", bio_i_1: \"{bio_i_1}\"")
    #  exit(-1)
    '''
    return {
      "curr_tok": tok,
      "prev_bio": bio_i_1,
    }

  def train(self, training_set=None,
            alg="GIS", trace=0, max_iter=10,
            timestamp=int(datetime.datetime.now().strftime('%m%d%H%M%S'))):
    assert training_set is not None, "training set is not defined!"
    self.training_set = training_set
    train_features = []
    for i, sample in enumerate(training_set):
      #if i % 100 == 0:
      #  print(i)
      toks, poss, bios = sample
      last_bio = "O"
      for item in zip(toks.split(), bios.split()):
        # item[0] is tok, item[1] is bio
        train_features.append(
          #(feature, tag)
          (self.gen_feature(item[0], last_bio), item[1])
        )
        last_bio = item[1]

    print("start to train MaxentClassifier")
    self.maxent = MaxentClassifier.train(train_features, algorithm=alg, trace=trace, max_iter=max_iter)
    print("train MaxentClassifier finish")

    if not os.path.exists("nltk_maxent"):
      os.makedirs("nltk_maxent")
    filepath = os.path.join("nltk_maxent", f"nltk_maxent_max_iter_{max_iter}_{timestamp}.pickle")
    with open(filepath, "wb") as fout:
      pickle.dump(self.maxent, fout)
    '''
    import pickle
    f = open('my_classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
    '''

  def load_model(self, model_path):
    with open(model_path, "rb") as fin:
      self.maxent = pickle.load(fin)

  def calc_prob(self, tok, bio_i, bio_i_1):
    pdist = self.maxent.prob_classify(self.gen_feature(tok, bio_i_1))
    return pdist.prob(bio_i)

  def calc_probs(self, tok, bio_i_1):
    pdist = self.maxent.prob_classify(self.gen_feature(tok, bio_i_1))
    return {
      tag : pdist.prob(tag) for tag in pdist.samples()
    }

  def predict(self, tok, bio_i_1):
    probs = self.calc_probs(tok, bio_i_1)
    tag = ""
    prob = 0
    for k, v in probs.items():
      if v > prob:
        prob = v
        tag = k
    return tag

  def valid_performance(self, validation_set):
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
      for item in zip(toks.split(), bios.split()):
        # item[0] is tok, item[1] is bio
        pred = self.predict(item[0], last_bio)
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
        #print(pred, item[1], end="; ")
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
  print(f"=== time used {time.time() - t0}")
  print("=== training maxent ===")
  memm = MEMMProb(None)
  memm.train(train, max_iter=10)
  print(f"=== time used {time.time() - t0}")
  print("done")
