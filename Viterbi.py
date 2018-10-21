#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 13:18:40 2018

@author: brucewxh
"""

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
    
    def _init(self, word, tag, prob_HMM, prob_MEMM, isHMM):
        self.word = word
        self.tag = tag
        self.word_size = len(word)
        self.tag_size = len(tag)
        self.score = [[0 for y in range(word_size)] for x in range(tag_size)]
        self.bptr = [[0 for y in range(word_size)] for x in range(tag_size)]
        if isHMM == true:
            self.isHMM = true
            self.prob_HMM = prob_HMM
        else:
            self.isHMM = false
            self.prob_MEMM = prob_MEMM
    
    def viterbi_alg():
        prob = self.prob_HMM if isHMM else self.prob_MEMM
        #initialization
        for i in range(self.tag_size):
            #score[i][0] = p(ti|)*p(wi|ti)
            score[i][0] = prob[tag[i]]
            bptr[i][0] = 0
        
        #iteration
        for t in range(1, word_size):
            for i in range(0, tag_size):
                local_max = 0
                curr = -1
                for j in range(0, tag_size):
                    #prob[tag[i]] = p(ti|tj)*p(wi|ti)
                    if(score[j][t - 1] * prob[tag[i]] >= max):
                        local_max = score[j][t - 1] * prob[tag[i]]
                        curr = j
                score[i][t] = local_max
                bptr[i][t] = curr
        
        #identify sequence
        res = ['' for i in range(word_size)]
        idx = 0
        tmp = 0
        for i in range(tag_size):
            if(score[i][word_size - 1] >= max):
                tmp = score[i][word_size - 1]
                idx = i
        
        res[word_size - 1] = tag[idx]
        for i in range(word_size - 2, -1):
            idx = bptr[idx][i + 1]
            res[i] = tag[idx]
        
        return res
            
                    
        
        
        