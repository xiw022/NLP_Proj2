# -*- coding: utf-8 -*-

from collections import Counter
import re

'''
  unk_mode:
    - "none"      : do not use "<unk>" token
    - "only_one"  : only count words appearing once as "<unk>"
    - "every_one" : for every word w_i, when it appears the first time, 
                    add 1 to #(<unk>) rather than #(w_i)
    - all other parameter will be interpreted as "none"
'''



class UnigramModel:
  '''
  token_cnt       := (dict) token : #(token)
  total_token_num := (int)  number of all tokens
  vocab           := (set)  words appeared in given token list
  vocab_size      := (int)  vocab size
  k               := (int)  smoothing parameter in add-k
  '''
  def __init__(self, tokens, k=0, unk_mode="none"):
    if not tokens or len(tokens) == 0:
      raise RuntimeError("constructing UnigramModel: empty stream of tokens")
    self.total_token_num = len(tokens)

    self.token_cnt = Counter(tokens)

    if unk_mode == "only_one":
      self.token_cnt["<unk>"] = 0
      tokens_to_delete = []
      for token, cnt in self.token_cnt.items():
        if cnt == 1:
          self.token_cnt["<unk>"] += 1
          tokens_to_delete.append(token)
      for token in tokens_to_delete:
        del self.token_cnt[token]
      if self.token_cnt["<unk>"] == 0:
        print("constructing UnigramModel: #(<unk>) is zero")
    elif re.match(r"every_(1|0\.\d*[1-9])", unk_mode):
      x = float(unk_mode[6:])
      self.token_cnt["<unk>"] = 0
      tokens_to_delete = []
      for token, _ in self.token_cnt.items():
        self.token_cnt["<unk>"] += x
        self.token_cnt[token] -= x
        if self.token_cnt[token] <= 0:
          tokens_to_delete.append(token)
      for token in tokens_to_delete:
        del self.token_cnt[token]
    self.vocab = set(self.token_cnt.keys())
    self.vocab_size = len(self.vocab)
    self.k = k
    self.kV = k * self.vocab_size

  def token_count(self, token):
    if token not in self.token_cnt:
      print("UnigramModel .token_count(token): token is not in the vocabulary, returning #(<unk>)")
      return self.token_cnt["unk"]
    return self.token_cnt[token]

  def calculate_prob(self, token, k=None):
    if token not in self.token_cnt:
      token = "unk"
    if not k:
      prob = (self.token_cnt[token] + self.k) / (self.total_token_num + self.kV)
    else:
      prob = (self.token_cnt[token] + k) / (self.total_token_num + k * self.vocab_size)

    return prob