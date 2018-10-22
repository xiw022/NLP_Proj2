#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
based on XW's version
"""

import data_reader
import memm_prob
import hmm_prob
import baseline
import time
import data_reader
import pickle

import math


TAGS = ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC", "O"]


def viterbi(predictor, sample,
            longer=False, pos=False,
            length=False, capital=False):

  #sample = data[0]
  toks, poss, bios = sample

  toks = toks.split()
  poss = poss.split()
  bios = bios.split()

  sample_len = len(toks)

  bptr = [{ tag : "" for tag in TAGS } for _ in range(sample_len)]

  #scores = { tag : -math.inf for tag in TAGS}

  cur_max_tag = "O"
  cur_max_prob = -math.inf

  # init
  tok_0 = toks[0]
  pos_0 = poss[0]
  scores = {tag: -math.inf for tag in TAGS}
  for t_i in TAGS:
    prob = predictor.calc_prob(tok_i=tok_0, bio_i=t_i, bio_i_1="O",
                              longer=longer, tok_i_1="",
                              pos=pos, pos_i=pos_0, pos_i_1="",
                              length=length, capital=capital)
    prob = math.log(prob)
    if prob > cur_max_prob:
      cur_max_prob = prob
      cur_max_tag = t_i
    if prob > scores[t_i]:
      bptr[0][t_i] = "O"
      scores[t_i] = prob

  # iteration
  tok_i_1 = tok_0
  pos_i_1 = pos_0
  for i, item in enumerate(zip(toks, poss)):
    if i == 0:
      continue
    tok_i, pos_i = item
    local_scores = { tag : -math.inf for tag in TAGS}

    probs = {}
    for t_i_1 in TAGS:
      probs[t_i_1] = predictor.calc_probs(tok_i=tok_i, bio_i_1=t_i_1,
                                         longer=longer, tok_i_1=tok_i_1,
                                         pos=pos, pos_i=pos_i, pos_i_1=pos_i_1,
                                         length=length, capital=capital)

    for t_i in TAGS:
      for t_i_1 in TAGS:
        prob = probs[t_i_1][t_i]
        # prob = predictor.calc_prob(tok_i=tok_i, bio_i=t_i, bio_i_1=t_i_1,
        #                             longer=longer, tok_i_1=tok_i_1,
        #                             pos=pos, pos_i=pos_i, pos_i_1=pos_i_1,
        #                             length=length, capital=capital)
        prob = math.log(prob)
        prob += scores[t_i_1] if scores[t_i_1] > -math.inf else 0
        if prob > cur_max_prob:
          cur_max_prob = prob
          cur_max_tag = t_i
        if prob > local_scores[t_i]:
          bptr[i][t_i] = t_i_1
          local_scores[t_i] = prob

    for tag in TAGS:
      scores[tag] = local_scores[tag]

    tok_i_1 = tok_i
    pos_i_1 = pos_i

  result = ["" for _ in range(sample_len)]
  # iteration
  # for i, tok_i in enumerate(toks):
  #   local_scores = { tag : -math.inf for tag in TAGS}
  #
  #   for t_i_1 in TAGS:
  #     probs = predictor.calc_probs(tok_i = tok_i, bio_i_1=t_i_1)
  #
  #     for tag in TAGS:
  #       if scores[t_i_1] + probs[tag] > cur_max_prob:
  #         cur_max_prob = scores[t_i_1] + probs[tag]
  #         cur_max_tag = tag
  #       if scores[t_i_1] + probs[tag] > local_scores[tag]:
  #         local_scores[tag] = scores[t_i_1] + probs[tag]
  #         bptr[i][tag] = t_i_1
  #
  #   for tag in TAGS:
  #     scores[tag] = local_scores[tag]
  #
  # result = ["" for _ in range(sample_len)]
  # back-pointing
  result[-1] = cur_max_tag
  for i in range(sample_len - 1, 0, -1):
    result[i-1] = bptr[i][result[i]]

  return result




#
# class Viterbi:
#     '''
#     word  := (list) all word
#     tag  := (list) all tag
#     word_size = (int) size of word
#     tag_size = (int) size of tag
#     prob_HMM = probability from HMM
#     prob_MEMM = probability from MEMM
#     score = (int[][]) viterbi score
#     bptr = (int[][]) back pointer
#     isHMM = (bool) whether use HMM
#     '''
#     
#     def __init__(self, hmm, memm, isHMM, train_set):
#         self.word = []
#         self.tag = []
#         self.res = []
#         if isHMM == True:
#             self.isHMM = True
#             self.hmm = hmm
#             for i, sample in enumerate(train_set):
#               toks, poss, bios = sample
#               for item in zip(toks.split(), bios.split(), poss.split()):
#                   self.word.append(item[0])
#                   self.tag.append(item[1])
#             #self.word = list(self.hmm.tag2tok.values())
#             #self.tag = list(self.hmm.tag2tok.keys())
#             print(self.word[0])
#             print(self.tag[0])
#             print(len(self.word))
#             print(len(self.tag))
#             #print(str(self.hmm.tag2tok))
#         else:
#             self.isHMM = False
#             self.memm = memm
#         
#         self.word_size = len(self.word)
#         self.tag_size = len(self.tag)
#         self.score = [[0 for y in range(self.word_size)] for x in range(self.tag_size)]
#         self.bptr = [[0 for y in range(self.word_size)] for x in range(self.tag_size)]
#     
#     def viterbi_alg(self):
#         prob = hmm if self.isHMM else memm
#         #initialization
#         for i in range(1, self.tag_size):
#             #score[i][0] = p(ti|)*p(wi|ti)
#             self.score[i][0] = prob.calc_prob(self.word[i], self.tag[i], self.tag[i - 1])
#             self.bptr[i][0] = 0
#         
#         #iteration
#         for t in range(1, self.word_size):
#             for i in range(0, self.tag_size):
#                 local_max = 0
#                 curr = -1
#                 for j in range(0, self.tag_size):
#                     #prob[tag[i]] = p(ti|tj)*p(wi|ti)
#                     if(self.score[j][t - 1] * prob.calc_prob(self.word[t], self.tag[i], self.tag[i - 1]) >= local_max):
#                         local_max = self.score[j][t - 1] * prob.calc_prob(self.word[t], self.tag[i], self.tag[i - 1])
#                         curr = j
#                 self.score[i][t] = local_max
#                 self.bptr[i][t] = curr
#         
#         #identify sequence
#         res = ['' for i in range(self.word_size)]
#         idx = 0
#         tmp = 0
#         for i in range(self.tag_size):
#             if(self.score[i][self.word_size - 1] >= local_max):
#                 tmp = self.score[i][self.word_size - 1]
#                 idx = i
#         
#         res[self.word_size - 1] = self.tag[idx]
#         for i in range(self.word_size - 2, -1):
#             idx = self.bptr[idx][i + 1]
#             res[i] = self.tag[idx]
#         
#         return res
#     
#     def valid_performance(self, validation_set,
#                             longer=False, pos=False,
#                             length=False, capital=False):
#         print("performance upon validation with"
#               f"{' longer' if longer else ''}"
#               f"{' pos' if pos else ''}"
#               f"{' length' if length else ''}"
#               f"{' capital' if capital else ''}.pickle")
#         total = 0
#         correct = 0
#         p_total = 0
#         p_correct = 0
#         r_total = 0
#         r_correct = 0
#         for i, sample in enumerate(validation_set):
#           #if i % 100 == 0:
#           #  print(i)
#           toks, poss, bios = sample
#           last_bio = "O"
#           last_tok = ""
#           last_pos = ""
#           #print(toks)
#           i = 0
#           pred = self.res
#           for item in zip(toks.split(), bios.split(), poss.split()):
#             # item[0] is tok, item[1] is bio, item[2] is pos
#             # p
#             if pred[i] != "O":
#               p_total += 1
#               if pred[i] == item[1]:
#                 p_correct += 1
#             # r
#             if item[1] != "O":
#               r_total += 1
#               if pred[i] == item[1]:
#                 r_correct += 1
#             # acc
#             if pred[i] == item[1]:
#               correct += 1
#             total += 1
#             #print(pred, item[1], end="; ")
#             last_bio = item[1]
#             last_tok = item[0]
#             last_pos = item[2]
#           #print(f"=== {i:>4d}: {correct/total} === \n")
#         p = p_correct/p_total
#         r = r_correct/r_total
#         return {
#           "p"  : p,
#           "r"  : r,
#           "f"  : 2*p*r/(p+r),
#           "acc": correct/total,
#         }
# 
# if __name__ == "__main__":
#   valid_ind_path = '''data_ind_split/simple_valid_portion_0.1_seed_1021122311.pickle'''
#   '''feature = length'''
#   timestamp = "1021122311"
#   reader = data_reader.DataReader()
#   reader.read_file()
#   train = reader.split_train_valid_by_valid_ind(valid_ind_path)
#   #train = ret["train"]
#   #valid = ret["valid"]
#   memm = ""
#   hmm = hmm_prob.HMMProb()
#   hmm.train(train)
#   v = Viterbi(hmm, memm, True, train)
#   print(v.viterbi_alg())
#             
#                     
#         
#         
#         