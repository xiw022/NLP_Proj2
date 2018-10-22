import numpy as np
import nltk
#from gensim.models import KeyedVectors
from nltk.classify import MaxentClassifier
import pickle
import datetime
import os



class MEMMProb:
  def __init__(self, word2vec=None, w2v_len=300, is_bin=True):
    self.training_set = None
    # if word2vec is None:
    #   self.w2v = None
    # else:
    #   self.w2v = KeyedVectors.load_word2vec_format(word2vec, binary=is_bin)
    self.w2v = None
    self.w2v_len = w2v_len
    self.maxent = None

  def gen_feature(self, tok_i, bio_i_1,
                  longer=False, tok_i_1=None,
                  pos=False, pos_i=None, pos_i_1=None,
                  length=False, capital=False):
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

    data = {
        "curr_tok": tok_i,
        "prev_bio": bio_i_1,
      }
    if pos:
      data["curr_pos"] = pos_i
    if length:
      data["curr_tok_len"] = len(tok_i)
    if capital:
      data["curr_tok_cap"] = sum(1 for c in tok_i if c.isupper())

    if longer:
      data["prev_tok"] = tok_i_1
      if pos:
        data["prev_pos"] = pos_i_1
      if length:
        data["prev_tok_len"] = len(tok_i_1)
      if capital:
        data["prev_tok_cap"] = sum(1 for c in tok_i_1 if c.isupper())
#      if length:
#        data["curr_tok_len"] = len(data["curr_tok"])
    ## NOT FINISHED YET !!!

    return data

  def train(self, training_set=None,
            alg="GIS", trace=0, max_iter=10,
            timestamp=int(datetime.datetime.now().strftime('%m%d%H%M%S')),
            longer=False, pos=False,
            length=False, capital=False):
    assert training_set is not None, "training set is not defined!"
    self.training_set = training_set
    train_features = []
    for i, sample in enumerate(training_set):
      #if i % 100 == 0:
      #  print(i)
      toks, poss, bios = sample
      prev_bio = "O"
      prev_tok = ""
      prev_pos = ""
      for item in zip(toks.split(), bios.split(), poss.split()):
        # item[0] is tok, item[1] is bio, item[2] is pos
        train_features.append(
          #(feature, tag)
          (self.gen_feature(item[0], prev_bio,
                            longer=longer, tok_i_1=prev_tok,
                            pos=pos, pos_i=item[2], pos_i_1=prev_pos,
                            length=length, capital=capital), item[1])
        )
        prev_pos = item[2]
        prev_bio = item[1]
        prev_tok = item[0]

    print("start to train MaxentClassifier")
    self.maxent = MaxentClassifier.train(train_features, algorithm=alg, trace=trace, max_iter=max_iter)
    print("train MaxentClassifier finish")

    if not os.path.exists("nltk_maxent"):
      os.makedirs("nltk_maxent")
    filepath = os.path.join("nltk_maxent",
                            f"nltk_maxent_max_iter_{max_iter}_{timestamp}"
                            f"{'_longer' if longer else ''}"
                            f"{'_pos' if pos else ''}"
                            f"{'_length' if length else ''}"
                            f"{'_capital' if capital else ''}.pickle")
    with open(filepath, "wb") as fout:
      pickle.dump(self.maxent, fout)
      print("dump model to file:", filepath)
    '''
    import pickle
    f = open('my_classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
    '''

  def load_model(self, model_path):
    with open(model_path, "rb") as fin:
      self.maxent = pickle.load(fin)

  def calc_prob(self, tok_i, bio_i, bio_i_1,
                  longer=False, tok_i_1=None,
                  pos=False, pos_i=None, pos_i_1=None,
                  length=False, capital=False):

    pdist = self.maxent.prob_classify(
      self.gen_feature(tok_i=tok_i, bio_i_1=bio_i_1,
                      longer=longer, tok_i_1=tok_i_1,
                      pos=pos, pos_i=pos_i, pos_i_1=pos_i_1,
                      length=length, capital=capital)
    )

    return pdist.prob(bio_i)

  def calc_probs(self, tok_i, bio_i_1,
                  longer=False, tok_i_1=None,
                  pos=False, pos_i=None, pos_i_1=None,
                  length=False, capital=False):

    pdist = self.maxent.prob_classify(
      self.gen_feature(tok_i=tok_i, bio_i_1=bio_i_1,
                  longer=longer, tok_i_1=tok_i_1,
                  pos=pos, pos_i=pos_i, pos_i_1=pos_i_1,
                  length=length, capital=capital)
    )

    return {
      tag : pdist.prob(tag) for tag in pdist.samples()
    }

  def predict(self, tok_i, bio_i_1,
              longer=False, tok_i_1=None,
              pos=False, pos_i=None, pos_i_1=None,
              length=False, capital=False):
    probs = self.calc_probs(tok_i=tok_i, bio_i_1=bio_i_1,
                            longer=longer, tok_i_1=tok_i_1,
                            pos=pos, pos_i=pos_i, pos_i_1=pos_i_1,
                            length=length, capital=capital)
    tag = ""
    prob = 0
    for k, v in probs.items():
      if v > prob:
        prob = v
        tag = k
    return tag

  def valid_performance(self, validation_set,
                        longer=False, pos=False,
                        length=False, capital=False):
    print("performance upon validation with"
          f"{' longer' if longer else ''}"
          f"{' pos' if pos else ''}"
          f"{' length' if length else ''}"
          f"{' capital' if capital else ''}.pickle")
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
      last_tok = ""
      last_pos = ""
      #print(toks)
      for item in zip(toks.split(), bios.split(), poss.split()):
        # item[0] is tok, item[1] is bio, item[2] is pos
        pred = self.predict(item[0], last_bio,
                            longer=longer, tok_i_1=last_tok,
                            pos=pos, pos_i=item[2], pos_i_1=last_pos,
                            length=length, capital=capital)
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
        last_tok = item[0]
        last_pos = item[2]
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
  memm.train(train, max_iter=10, length=True, capital=True)
  print(f"=== time used {time.time() - t0}")
  print("done")
