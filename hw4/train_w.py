#!/usr/bin/env python
import sys
import argparse
from collections import defaultdict
from utils import read_ttable
from pprint import pprint

def dot(w, v):
    s = 0.0
    for k in set(w.keys()) & set(v.keys()):
        s += w[k] * v[k]
    return s

def sub(f1, f2):
    """ f1 - f2 """

    result = defaultdict(float)
    for item in f1:
        result[item] = f1[item] - f2[item]
    return result

def sub_value(w, alpha):
    result = defaultdict(float)
    for item in w:
        result[item] = w[item] - alpha
    return result

def mult_value(w, alpha):
    for item in w:
        w[item] = w[item] * alpha
    return w
    


parser = argparse.ArgumentParser()
parser.add_argument('--train_input', '-i', default='data/train.input')
parser.add_argument('--train_refs', '-r', default='data/train.refs')
parser.add_argument('--dev_input', '-a', default="data/dev+test.input")
parser.add_argument('--dev_refs', '-b', default="data/dev+test.refs")
parser.add_argument('--ttable', '-t', default='data/ttable')
args = parser.parse_args()

translation_table = read_ttable(args.ttable)

# Initial Weights
weights = {'log_prob_tgs': 0.7,
           "log_prob_sgt": 0.1,
           "log_lex_prob_tgs": 0.1,
           "log_lex_prob_sgt": 0.1}

def loss(weights):
    CORPUS_LOSS = 0.0
    for line,ref in zip(open(args.train_input), open(args.train_refs)):
        ref = ref.decode('utf-8').strip()
        left_context, phrase, right_context = [part.strip() for part in line.decode('utf-8').strip().split('|||')]
        
        ref_features = translation_table[phrase][ref]
        INSTANCE_LOSS = 0.0
        for target,features in translation_table[phrase].iteritems():
            if target != ref:
                result = sub(ref_features, features)

                gamma = 1.0
                term = max(0.0, gamma - dot(weights,result))
                INSTANCE_LOSS += term
    
        CORPUS_LOSS += INSTANCE_LOSS
    return CORPUS_LOSS


def train(weights):
    for line,ref in zip(open(args.train_input), open(args.train_refs)):
        ref = ref.decode('utf-8').strip()
    
        left_context, phrase, right_context = [part.strip() for part in line.decode('utf-8').strip().split('|||')]
        #candidates = [target for target, features in sorted(translation_table[phrase].iteritems(), key=lambda (t, f): dot(f, weights), reverse=True)]
    
        """
        print left_context
        print phrase
        print right_context
        print ref
        """
        
        ref_features = translation_table[phrase][ref]
        for target,features in translation_table[phrase].iteritems():
            if target != ref:
                result = sub(ref_features, features)
                
                gamma = 1.0
                term = max(0.0, gamma - dot(weights,result))
                if term > 0.0:
                    gradient = sub(features, ref_features)

                    alpha = 0.1
                    gradient = mult_value(gradient, alpha)
                    weights = sub(weights, gradient)
    return weights


def test(weights):
    for line in open(args.input):
        left_context, phrase, right_context = [part.strip() for part in line.decode('utf-8').strip().split('|||')]
        candidates = [target for target, features in sorted(translation_table[phrase].iteritems(), key=lambda (t, f): dot(f, weights), reverse=True)]
        print ' ||| '.join(candidates).encode('utf-8')
        
                



CORPUS_LOSS = loss(weights)

print "WEIGHTS:", weights
print "STARTING LOSS:", CORPUS_LOSS
for i in xrange(5):
    weights = train(weights)
    CORPUS_LOSS = loss(weights)

    print
    print i, "WEIGHTS:", weights
    print i, "CORPUS LOSS:", CORPUS_LOSS
