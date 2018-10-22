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


class BigramModel:
  '''
  token_cnt  := (dict) token : #(token)
  token_pair_cnt  := (dict) (tok1, tok2) : #(tok1, tok2)
  vocab           := (set)  words appeared in given token list
  vocab_size      := (int)  vocab size
  k               := (int)  smoothing parameter in add-k
  '''


  def __init__(self, tokens, k=0, unk_mode="none"):
    tokens = list(tokens)

    def generate_word_pair(tokens):
      for i in range(len(tokens) - 1):
        yield (tokens[i], tokens[i + 1])

    def init_unk_only_one():
      # get token count
      self.token_cnt = Counter(tokens)
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
      # get token pair count
      self.token_pair_cnt = dict()
      for prev, curr in generate_word_pair(tokens):
        if prev not in self.token_cnt:
          # since every "appeared once" tokens are deleted from token_cnt
          # if token not in token_cnt, it should be replaced by "<unk>"
          prev = "<unk>"
        if curr not in self.token_cnt:
          curr = "<unk>"
        self.token_pair_cnt[(prev, curr)] = self.token_pair_cnt.get((prev, curr), 0) + 1

    def init_unk_every_x(x):
      # get token count
      self.token_cnt = Counter(tokens)
      self.token_cnt["<unk>"] = 0
      tokens_to_delete = []
      for token in self.token_cnt.keys():
        self.token_cnt["<unk>"] += x
        self.token_cnt[token] -= x
        if self.token_cnt[token] <= 0:
          tokens_to_delete.append(token)
      for token in tokens_to_delete:
        del self.token_cnt[token]
      # get token pair count
      appeared = set()
      self.token_pair_cnt = dict()
      for prev, curr in generate_word_pair(tokens):
        #print(prev, curr, appeared)
        prev_unk = None
        curr_unk = None
        if prev not in appeared:
          # if prev has not appeared before, it should be replaced by "<unk>"
          appeared.add(prev)
          prev_unk = "<unk>"
        if curr not in appeared:
          appeared.add(curr)
          curr_unk = "<unk>"
        #print(prev, curr, appeared)
        self.token_pair_cnt[(prev, curr)] = self.token_pair_cnt.get((prev, curr), 0) + 1


    if not tokens or len(tokens) == 0:
      raise RuntimeError("constructing BigramModel: empty stream of tokens")
    if len(tokens) == 1:
      raise RuntimeError("constructing BigramModel: token stream too short (length=1)")

    if unk_mode == "only_one":
      init_unk_only_one()
    elif re.match(r"every_(1|0\.\d*[1-9])", unk_mode):
      x = float(unk_mode[6:])
      init_unk_every_x(x)
    else:
      self.token_cnt = Counter(tokens)
      self.token_pair_cnt = dict()
      for prev, curr in generate_word_pair(tokens):
        self.token_pair_cnt[(prev, curr)] = self.token_pair_cnt.get((prev, curr), 0) + 1


    self.vocab = set(self.token_cnt.keys())
    self.vocab_size = len(self.vocab)

    self.k = k
    self.kV = k * self.vocab_size


  def calculate_prob(self, prev, curr, k=None):
    if prev not in self.vocab:
      prev = "<unk>"
    if curr not in self.vocab:
      curr = "<unk>"
    if not k:
      prob = (self.token_pair_cnt.get((prev, curr), 0) + self.k) / (self.token_cnt[prev] + self.kV)
    else:
      prob = (self.token_pair_cnt.get((prev, curr), 0) + k) / (self.token_cnt[prev] + k * self.vocab_size)
    return prob