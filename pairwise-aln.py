#!/usr/bin/env python3
import re, csv, sys, os, glob, warnings, itertools
from math import ceil
from optparse import OptionParser
from operator import itemgetter
import numpy as np
import pandas as pd

parser=OptionParser(description='global pairwise alignment of two input sequences and return the alignment result')
parser.add_option('-i','--input','--in',default='.',help='a .txt file contain pairs of input sequences, sequence separated by `,`; pairs seperated by `\\n`')
parser.add_option('-o', default='algn.txt', help="where to output the alignment result [default: %default]")
parser.add_option('-a','--match_score', default=1, type='int', help="score of match [default: %default]")
parser.add_option('-m', '--mismatch_score', default=-1, type='int', help="score of mismatch [default: %default]")
parser.add_option('-g', '--gap_score', default=-1, type='int', help="score of gap (using the same score for gap opening and extension)  [default: %default]")

(opts, args)=parser.parse_args()

pair_lst = [] #store the pair(s) of sequences
if (os.path.isfile(opts.input)):
    try:
        fin = open(opts.input,'r')
        for line in fin:
            pair_lst.append(line.split(','))
    except IOError:
        print("[Align_Error 1]: input file, %s, doesn;t exist; expect to read a .txt with `,`seperated pair(s) of sequences"%(opts.input))
    pair_num = len(pair_lst)
else:
    ## Check that opts.input directory exists
    if not os.path.isdir(opts.input):
      parser.print_help()
      print(" ")
      print("[Align_Error 2]: directory '%s' not found!" % (opts.input))
      sys.exit(1)

#####
## Checks whether sequences valid
#####

#.txt 有多行输入 未实现
class Problem:
    def __init__(self, input_lst,match_score,mismatch_score,gap_score):
        self.pair = 0 # for cnt of multiple pairs
        self.path = 0
        self.indicator = '' # for visulization
        self.alignment_score = 0
        self.match_score = match_score
        self.mismatch_score =mismatch_score
        self.gap_score = gap_score
        self.align_jobs = len(input_lst)
        self.x = [i[0] for i in input_lst][self.pair]
        self.output1 = self.x
        self.row = len(self.x)
        self.y = [i[1] for i in input_lst][self.pair]
        self.output2 = self.y
        self.col = len(self.y)
        self.current_state = [1,1]
        self.end_state = [self.row-1,self.col-1]
        self.strategy_space = np.zeros(shape = (self.row,self.col))
        self.strategy_space[0,0] = 1
        print(self.row,self.col)

        location = []
        for i in range(self.row):
            for j in range(self.col):
                location.append((i,j))
        self.successor={i:[] for i in location}
        self.successor[(0,0)]=[(1,1)]
        for i in range(1,self.col):
            self.strategy_space[0,i] = self.strategy_space[0,i-1] + self.gap_score
        for i in range(1,self.row):
            self.strategy_space[i,0] = self.strategy_space[i-1,0] + self.gap_score
        while self.current_state!=self.end_state:
            self.bellman_equation()
        with open(opts.o, 'a') as outfile: 
            for row in self.strategy_space: 
                outfile.write(',\t'.join([str(num) for num in row])) 
                outfile.write('\n') 
            for i in self.trace_back():
                outfile.write(str(i))
        self.output()
        return

    def bellman_equation(self):
        i = self.current_state[0]
        j = self.current_state[1]
        # print(self.strategy_space[i+1,j])
        # print(self.strategy_space)
        if self.current_state[0] == self.row - 1:
            # print("======")
            score = max(self.gap_score+self.strategy_space[1,j],self.gap_score+self.strategy_space[0,j+1],self.strategy_space[0,j]+self.similarity_score(1,j+1))
            if score == self.gap_score+self.strategy_space[1,j]:
                self.successor[(1,j)].append((1,j+1))
            if score == self.gap_score+self.strategy_space[0,j+1]:
                self.successor[(0,j+1)].append((1,j+1))
            if score == self.strategy_space[0,j]+self.similarity_score(1,j+1):
                self.successor[(0,j)].append((1,j+1))
            self.strategy_space[1,j+1] = score
            if self.current_state[1] < self.col - 1:
                self.current_state[1] += 1
                self.current_state[0] = 1
            else:
                # print('end state reached',self.current_state)
                return
        else:
            score = max(self.gap_score+self.strategy_space[i+1,j-1],self.gap_score+self.strategy_space[i,j],self.strategy_space[i,j-1]+self.similarity_score(i+1,j))

            if score == self.gap_score+self.strategy_space[i+1,j-1]:
                self.successor[(i+1,j-1)].append((i+1,j))
            if score == self.gap_score+self.strategy_space[i,j]:
                self.successor[(i,j)].append((i+1,j))
            if score == self.strategy_space[i,j-1]+self.similarity_score(i+1,j):
                self.successor[(i,j-1)].append((i+1,j))
            self.strategy_space[i+1,j] = score
            self.current_state[0]+=1
        # print(self.strategy_space)
        return

    def similarity_score(self,i,j):
        a = self.x[i]
        b = self.y[j]
        if a.upper()==b.upper():
            return self.match_score
        else:
            return self.mismatch_score

    def find_key(self,value):
        hold = []
        for i in list(self.successor.keys()):
            # print(i)
            tmp = self.successor[i]
            if value in tmp:
                hold.append(i)
        max_index = hold.index(max(hold, key=lambda x: self.strategy_space[x[0],x[1]]))
        return hold[max_index]


    def trace_back(self):
        self.path = [tuple(self.end_state),self.find_key(tuple(self.end_state))]
        # print(path)

        value = self.find_key(tuple(self.end_state))
        # print(value)
        # print(self.strategy_space)
        while value != (0,0) and value != (1,0) and value != (0,1):
            # print(value)
            value = self.find_key(value)
            self.path.append(value)
        self.path.append((0,0))
        self.path.reverse()
        # print(self.path)
        return self.path

    @staticmethod
    def _insert(raw,idx,string):
        str_cnt = 0
        for i in range(len(raw)):
            if raw[i].isalpha():
            # if isinstance(raw[i],str):
                str_cnt += 1
                if idx == str_cnt - 1:
                    left = raw[:str_cnt]
                    right = raw[str_cnt:]
                    tmp = [left,right]
                    new = string.join(tmp)
                    return new
        return raw

    def output(self):
        for i in range(len(self.path)-1):
            current = self.path[i]
            successor = self.path[i+1]
            y_diff = successor[1]-current[1]
            x_diff = successor[0]-current[0]
            if (y_diff * x_diff == 0):
                self.alignment_score += self.gap_score
                if (y_diff == 1):
                    # vertical gap 
                    self.output1 = Problem._insert(self.output1,i,'-')
                elif (x_diff == 1):
                    # horizontal gap
                    self.output2 = Problem._insert(self.output2,i,'-')
            else:            
                #diagonal
                if self.x[successor[0]]==self.y[successor[1]]:
                    # match
                    self.alignment_score += self.match_score
                else:
                    # mismatch
                    self.alignment_score += self.mismatch_score
        # print(self.output1,self.output2)
        for i in range(max(len(self.output1),len(self.output2))):
            # print(len(self.output1),len(self.output2))
            
            if (self.output1[i].isalpha()and self.output2[i].isalpha()):
                if self.output1[i]==self.output2[i]:
                    self.indicator += '|'
                else:
                    self.indicator += '*'
            else:
                self.indicator += ' '
        with open(opts.o, 'a') as outfile: 
            print("Alignment score: {}\nOptimal alignment:\n{}\n{}\n{}".format(self.alignment_score,self.output1,self.indicator,self.output2))
            outfile.write("Alignment score: {}\nOptimal alignment:\n{}\n{}\n{}".format(self.alignment_score,self.output1,self.indicator,self.output2))
        return

    
Problem(pair_lst,opts.match_score,opts.mismatch_score,opts.gap_score)