#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 13:18:40 2018

@author: brucewxh
"""
import data_reader
import memm_prob
import hmm_prob
import baseline
import time
import data_reader
import pickle

class Viterbi:
    '''
    word  := (list) all word
    tag  := (list) all tag
    word_size = (int) size of word
    tag_size = (int) size of tag
    prob_HMM = probability from HMM
    prob_MEMM = probability from MEMM
    score = (int[][]) viterbi score
    bptr = (int[][]) back pointer
    isHMM = (bool) whether use HMM
    '''
    
    def __init__(self, hmm, memm, isHMM, train_set):
        self.word = []
        self.tag = []
        self.res = []
        if isHMM == True:
            self.isHMM = True
            self.hmm = hmm
            for i, sample in enumerate(train_set):
              toks, poss, bios = sample
              for item in zip(toks.split(), bios.split(), poss.split()):
                  self.word.append(item[0])
                  self.tag.append(item[1])
            #self.word = list(self.hmm.tag2tok.values())
            #self.tag = list(self.hmm.tag2tok.keys())
            print(self.word[0])
            print(self.tag[0])
            print(len(self.word))
            print(len(self.tag))
            #print(str(self.hmm.tag2tok))
        else:
            self.isHMM = False
            self.memm = memm
        
        self.word_size = len(self.word)
        self.tag_size = len(self.tag)
        self.score = [[0 for y in range(self.word_size)] for x in range(self.tag_size)]
        self.bptr = [[0 for y in range(self.word_size)] for x in range(self.tag_size)]
    
    def viterbi_alg(self):
        prob = hmm if self.isHMM else memm
        #initialization
        for i in range(1, self.tag_size):
            #score[i][0] = p(ti|)*p(wi|ti)
            self.score[i][0] = prob.calc_prob(self.word[i], self.tag[i], self.tag[i - 1])
            self.bptr[i][0] = 0
        
        #iteration
        for t in range(1, self.word_size):
            for i in range(0, self.tag_size):
                local_max = 0
                curr = -1
                for j in range(0, self.tag_size):
                    #prob[tag[i]] = p(ti|tj)*p(wi|ti)
                    if(self.score[j][t - 1] * prob.calc_prob(self.word[t], self.tag[i], self.tag[i - 1]) >= local_max):
                        local_max = self.score[j][t - 1] * prob.calc_prob(self.word[t], self.tag[i], self.tag[i - 1])
                        curr = j
                self.score[i][t] = local_max
                self.bptr[i][t] = curr
        
        #identify sequence
        res = ['' for i in range(self.word_size)]
        idx = 0
        tmp = 0
        for i in range(self.tag_size):
            if(self.score[i][self.word_size - 1] >= local_max):
                tmp = self.score[i][self.word_size - 1]
                idx = i
        
        res[self.word_size - 1] = self.tag[idx]
        for i in range(self.word_size - 2, -1):
            idx = self.bptr[idx][i + 1]
            res[i] = self.tag[idx]
        
        return res
    
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
          i = 0
          pred = self.res
          for item in zip(toks.split(), bios.split(), poss.split()):
            # item[0] is tok, item[1] is bio, item[2] is pos
            # p
            if pred[i] != "O":
              p_total += 1
              if pred[i] == item[1]:
                p_correct += 1
            # r
            if item[1] != "O":
              r_total += 1
              if pred[i] == item[1]:
                r_correct += 1
            # acc
            if pred[i] == item[1]:
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
  valid_ind_path = '''data_ind_split/simple_valid_portion_0.1_seed_1021122311.pickle'''
  '''feature = length'''
  timestamp = "1021122311"
  reader = data_reader.DataReader()
  reader.read_file()
  train = reader.split_train_valid_by_valid_ind(valid_ind_path)
  #train = ret["train"]
  #valid = ret["valid"]
  memm = ""
  hmm = hmm_prob.HMMProb()
  hmm.train(train)
  v = Viterbi(hmm, memm, True, train)
  print(v.viterbi_alg())
            
                    
        
        
        