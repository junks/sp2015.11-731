#!/usr/bin/env python
import sys
import argparse
from collections import defaultdict
from utils import read_ttable
from pprint import pprint


def dot(w, v):
    s = 0.0
    for k in set(w.keys()) & set(v.keys()):
        s += w.get(k,0) * v.get(k,0)
    return s

def sub(f1, f2):
    """ f1 - f2 """
    result = defaultdict(float)
    for k in set(f1.keys()) & set(f2.keys()):
        value = f1.get(k,0) - f2.get(k,0)
        if value != 0:
            result[k] = value
    return result

def mult_value(w, alpha):
    for item in w:
        w[item] = w[item] * alpha
    return w


def output_predictions(predictions):
    f = open("output.txt", "w")
    for prediction in predictions:
        f.write(prediction+"\n")
    f.close()


parser = argparse.ArgumentParser()
parser.add_argument('--train_input', '-i', default='data/train.input')
parser.add_argument('--train_refs', '-r', default='data/train.refs')
parser.add_argument('--dev_input', '-a', default="data/dev+test.input")
parser.add_argument('--dev_refs', '-b', default="data/dev.refs")
parser.add_argument('--ttable', '-t', default='data/ttable')
args = parser.parse_args()

translation_table = read_ttable(args.ttable)

# Initial Weights
weights = {'log_prob_tgs': 0.9,
           "log_prob_sgt": 0.1,
           "log_lex_prob_tgs": 0.0,
           "log_lex_prob_sgt": 0.0}

lefts, phrases, rights, refs = [], [], [], []
for line,ref in zip(open(args.train_input), open(args.train_refs)):
    ref = ref.decode("utf-8").strip()
    left_context, phrase, right_context = [part.strip() for part in line.decode('utf-8').strip().split('|||')]
    lefts.append(left_context)
    phrases.append(phrase)
    rights.append(right_context)
    refs.append(ref)


def features(left,phrase,right,ref):
    ref_features = translation_table[phrase][ref]

    prev = ""
    if left:
        prev = left.split()[-1]

    next = ""
    if right:
        next = right.split()[0]
    src = phrase
    tgt = ref

    base = "src:"+src+"_tgt:"+tgt
    prevfeature = base +"_prev:"+prev
    nextfeature = base +"_next:"+next
    prevnextfeatures = {prevfeature:1.0, nextfeature:1.0}

    features = dict(ref_features, **prevnextfeatures)
    return features


def loss(w):
    CORPUS_LOSS = 0.0
    for left,phrase,right,ref in zip(lefts,phrases,rights,refs):
        #ref_features = translation_table[phrase][ref]
        ref_features = features(left,phrase,right,ref)

        for target,target_features in translation_table[phrase].iteritems():
            if target != ref:
                target_features = features(left,phrase,right,target)

                result = sub(ref_features, target_features)

                gamma = 1.0
                term = max(0.0, gamma - dot(w,result))
                CORPUS_LOSS += term
    return CORPUS_LOSS


def train(w):
    for left,phrase,right,ref in zip(lefts,phrases,rights,refs):
        #candidates = [target for target, features in sorted(translation_table[phrase].iteritems(), key=lambda (t, f): dot(f, w), reverse=True)]
        
        #ref_features = translation_table[phrase][ref]
        ref_features = features(left,phrase,right,ref)

        for target,target_features in translation_table[phrase].iteritems():
            if target != ref:
                target_features = features(left,phrase,right,target)

                result = sub(ref_features, target_features)
                
                gamma = 1.0
                term = max(0.0, gamma - dot(w,result))
                if term > 0.0:
                    alpha = -0.01
                    gradient = mult_value(result, alpha)
                    w = sub(w, gradient)
        #sys.stderr.write(".")
    #sys.stderr.write("\n")
    return w


def test(w):
    predictions = []
    for line in open(args.dev_input):
        left_context, phrase, right_context = [part.strip() for part in line.decode('utf-8').strip().split('|||')]
        candidates = [target for target, features in sorted(translation_table[phrase].iteritems(), key=lambda (t, f): dot(f, w), reverse=True)]
        prediction = ' ||| '.join(candidates).encode('utf-8')
        
        predictions.append(prediction)
    return predictions

def grade(predictions):
    score = 0.0
    sentence_count = 0

    for prediction, ref in zip(predictions, open(args.dev_refs)):
        prediction = [item.strip() for item in prediction.decode('utf-8').strip().split("|||")]
        ref = ref.decode('utf-8').strip()
        
        #print ref, ref in prediction
        
        found = False
        for i, hyp in enumerate(prediction):
            if hyp == ref:
                score += 1.0 / (i+1)

                break
        sentence_count += 1 

    return score / sentence_count


CORPUS_LOSS = loss(weights)
predictions = test(weights)
SCORE = grade(predictions)

ITERATIONS  = 1

print "WEIGHTS:", weights
print "STARTING LOSS:", CORPUS_LOSS
print "STARTING SCORE:", SCORE


for i in xrange(ITERATIONS):
    weights = train(weights)
    CORPUS_LOSS = loss(weights)
    predictions = test(weights)
    SCORE = grade(predictions)

    print
    print i, "WEIGHTS:", weights
    print i, "CORPUS LOSS:", CORPUS_LOSS
    print i, "SCORE:", SCORE
"""
"""
output_predictions(predictions)
