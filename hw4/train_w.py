#!/usr/bin/env python
import sys, time
import argparse
from collections import defaultdict
from utils import *
from pprint import pprint
from iterview import *
from interning import *

def dot(w, v):
    s = 0.0
    for k in set(w.keys()) & set(v.keys()):
        s += w[k] * v[k]
    return s

def sub(f1, f2):
    """ f1 - f2 """
    for k in f2.keys():
        value = f1.get(k,0.0) - f2[k]
        if value != 0.0:
            f1[k] = value
    return f1

def mult_value(w, alpha):
    for item in w:
        w[item] = w[item] * alpha
    return w

def output_predictions(predictions):
    sys.stderr.write("OUTPUTING PREDICTIONS\n\n")
    
    f = open("output.txt", "w")
    for prediction in predictions:
        prediction = " ||| ".join(prediction).encode('utf-8')
        f.write(prediction+"\n")
    f.close()


parser = argparse.ArgumentParser()
parser.add_argument('--train_input', '-i', default='data/train.input')
parser.add_argument('--train_refs', '-r', default='data/train.refs')
parser.add_argument('--dev_input', '-a', default="data/dev+test.input")
parser.add_argument('--dev_refs', '-b', default="data/dev.refs")
parser.add_argument('--ttable', '-t', default='data/ttable')

parser.add_argument("--train_parses", '-tp', default="data/train.parses")
parser.add_argument("--dev_parses", '-dp', default="data/dev+test.parses")

args = parser.parse_args()

# Initial Weights
weights = {'log_prob_tgs': 0.9,
           "log_prob_sgt": 0.1,
           "log_lex_prob_tgs": 0.0,
           "log_lex_prob_sgt": 0.0}
#internedWeights = {interning.intern(item):weights[item] for item in weights}
#weights = internedWeights

translation_table = read_ttable(args.ttable)


sys.stderr.write("READING PARSES\n")
start = time.time()
training_trees = []
for tree in read_dep_trees(args.train_parses):
    training_trees.append(tree)
devtest_trees = []
for tree in read_dep_trees(args.dev_parses):
    devtest_trees.append(tree)
end = time.time()
sys.stderr.write("FINISHED READING PARSES ("+str(end-start)+")\n")





PREFIX = 6000

i = 0
lefts, phrases, rights, refs = [], [], [], []
for line,ref in zip(open(args.train_input), open(args.train_refs)):

    if i > PREFIX:
        break
    ref = ref.decode("utf-8").strip()
    left_context, phrase, right_context = [part.strip() for part in line.decode('utf-8').strip().split('|||')]
    lefts.append(left_context)
    phrases.append(phrase)
    rights.append(right_context)
    refs.append(ref)
    i += 1


dev_lefts, dev_phrases, dev_rights = [], [], []
for line in open(args.dev_input):
    left_context, phrase, right_context = [part.strip() for part in line.decode('utf-8').strip().split('|||')]
    dev_lefts.append(left_context)
    dev_phrases.append(phrase)
    dev_rights.append(right_context)

dev_refs = []
for line in open(args.dev_refs):
    line = line.decode('utf-8').strip()
    dev_refs.append(line)


def features(left,phrase,right,ref,tree):
    #resultfeatures = {interning.intern(item):translation_table[phrase][ref][item] for item in translation_table[phrase][ref]}
    resultfeatures = {item:translation_table[phrase][ref][item] for item in translation_table[phrase][ref]}

    i = 0
    contextcut = 3
    wordcut = 3

    src = phrase[:wordcut]
    tgt = ref[:wordcut]
    base = "src:"+src+"_tgt:"+tgt

    prev_tag = ""
    prevprev_tag = ""
    if left:
        left = left.split()
        i = len(left)

        prev_tag = tree.tags[i-1]
        if len(left) > 1:
            prevprev_tag = tree.tags[i-2]

        for word in left:
            word = word[:contextcut]
            word_feature = base+"_context:"+word
            resultfeatures[word_feature] = 1.0

    next_tag = ""
    nextnext_tag = ""
    if right:
        right = right.split()
        next_tag = tree.tags[i+1]

        if len(right) > 1:
            nextnext_tag = tree.tags[i+1]

        for word in right:
            word = word[:contextcut]
            word_feature = base+"_context:"+word
            resultfeatures[word_feature] = 1.0
    parent,relation = tree.parents[i]
    parent_tag = "ROOT"
    if not parent is None:
        parent_tag = tree.tags[parent]

    #prevfeature = base +"_prev:"+prev_word
    #nextfeature = base +"_next:"+next_word

    #prevprevfeature = base + "_prevprev:"+prevprev_word
    #nextnextfeature = base + "_nextnext:"+nextnext_word

    #prevtag_feature = base+"_prevtag:"+prev_tag
    #prevprevtag_feature = base+"_prevprevtag:"+prevprev_tag
    #nexttag_feature = base+"_nexttag:"+next_tag
    #nextnexttag_feature = base+"_nextnexttag:"+nextnext_tag
    
    #parent_feature = base+"_parent:"+parent_tag+"_relation:"+relation

    """resultfeatures[interning.intern(prevfeature)] = 1.0
    resultfeatures[interning.intern(nextfeature)] = 1.0
    resultfeatures[interning.intern(prevprevfeature)] = 1.0
    resultfeatures[interning.intern(nextnextfeature)] = 1.0
    """
    #resultfeatures[prevtag_feature] = 1.0
    #resultfeatures[nexttag_feature] = 1.0
    #resultfeatures[parent_feature] = 1.0

    return resultfeatures

#def internize(f):
#    return {interning.intern(item):f[item] for item in f}


def loss(w):
    sys.stderr.write("CALCULATING CORPUS LOSS\n")
    start = time.time()
    
    CORPUS_LOSS = 0.0
    i = 0
    for _ in iterview(xrange(len(refs))):
        left,phrase,right,ref,tree = lefts[i],phrases[i],rights[i],refs[i],training_trees[i]
        ref_features = features(left,phrase,right,ref,tree)

        for target,_ in translation_table[phrase].iteritems():
            if target != ref:
                target_features = features(left,phrase,right,target,tree)

                result = sub(ref_features, target_features)

                gamma = 1.0
                term = max(0.0, gamma - dot(w,result))
                CORPUS_LOSS += term
        i += 1

    end = time.time()
    sys.stderr.write("FINISHED CORPUS LOSS: ("+str(end-start)+")\n\n")
    return CORPUS_LOSS


def train(w):
    sys.stderr.write("BEGIN TRAIN\n")
    start = time.time()

    #printonce = True
    i = 0
    for _ in iterview(xrange(len(refs))):
        left = lefts[i]
        phrase = phrases[i]
        right = rights[i]
        ref = refs[i]

        tree = training_trees[i]

        ref_features = features(left,phrase,right,ref,tree)

        for target,target_features in translation_table[phrase].iteritems():
            if target != ref:
                target_features = features(left,phrase,right,target,tree)
                result = sub(ref_features, target_features)
                
                gamma = 1.0
                term = max(0.0, gamma - dot(w,result))
                if term != 0.0:
                    alpha = -0.001
                    gradient = mult_value(result, alpha)
                    w = sub(w, gradient)
        i += 1
    end = time.time()
    sys.stderr.write("FINISHED TRAIN: ("+str(end-start)+")\n\n")
    return w


def test(w):
    sys.stderr.write("BEGIN TEST\n")
    start = time.time()

    predictions = []
    i = 0
    for _ in iterview(xrange(len(dev_phrases))):
        #for _ in xrange(len(dev_phrases)):
        left_context,phrase,right_context,tree = dev_lefts[i],dev_phrases[i],dev_rights[i],devtest_trees[i]
        candidates = [target for score,target in 
                      sorted([(dot(w,f),target) for f,target in 
                              [(features(left_context, phrase, right_context, target,tree),target) for target in 
                               [target for target,_ in translation_table[phrase].iteritems()]]], reverse=True)]
                      
        #prediction = ' ||| '.join(candidates).encode('utf-8')
        predictions.append(candidates)
        i += 1
    end = time.time()
    sys.stderr.write("FINISHED TEST ("+str(end-start)+")\n")
    return predictions

def grade(predictions):
    sys.stderr.write("BEGIN GRADING\n")
    start = time.time()

    score = 0.0
    sentence_count = 0

    for prediction,ref in zip(predictions, dev_refs):
        for i, hyp in enumerate(prediction):
            if hyp == ref:
                score += 1.0 / (i+1)
                break
        sentence_count += 1 

    end = time.time()
    sys.stderr.write("FINISHED GRADING: ("+str(end-start)+")\n\n")
    return score / sentence_count



# Grades the existing output.txt file
def grade_file():
    sys.stderr.write("BEGIN GRADING FILE\n")
    start = time.time()

    score = 0.0
    sentence_count = 0

    for prediction, ref in zip(open("output.txt","r"), open(args.dev_refs)):
        prediction = [item.strip() for item in prediction.decode('utf-8').strip().split("|||")]
        ref = ref.decode('utf-8').strip()
        
        for i, hyp in enumerate(prediction):
            if hyp == ref:
                score += 1.0 / (i+1)
                break
        sentence_count += 1 

    end = time.time()
    sys.stderr.write("FINISHED GRADING FILE: ("+str(end-start)+")\n\n")
    return score / sentence_count



ITERATIONS  = 100

CORPUS_LOSS = loss(weights)
predictions = test(weights)
SCORE = grade(predictions)
FILE_SCORE = grade_file()

BEST_SCORE = max(SCORE, FILE_SCORE)
#output_predictions(predictions)

print "WEIGHTS:", weights
print "STARTING LOSS:", CORPUS_LOSS
print "STARTING SCORE:", SCORE
print "BEST SCORE:", BEST_SCORE




print


#printOnce = True
for i in xrange(ITERATIONS):
    print "ITERATION:", i
    start = time.time()
    weights = train(weights)
    #CORPUS_LOSS = loss(weights)
    predictions = test(weights)
    SCORE = grade(predictions)
    FILE_SCORE = grade_file()
    
    if SCORE > BEST_SCORE and SCORE > FILE_SCORE:
        BEST_SCORE = SCORE
        output_predictions(predictions)

    """
    if printOnce:
        pprint(weights, open("weights.txt","w"))
        printOnce = False
        """
    print
    print "LENGTH OF WEIGHTS:", len(weights)
    #print i, "CORPUS LOSS:", CORPUS_LOSS
    print i, "SCORE:", SCORE
    end = time.time()
    print "TIME ELAPSED:", end-start
    print


"""
"""
#output_predictions(predictions)


CORPUS_LOSS = loss(weights)
print "CORPUS LOSS:", CORPUS_LOSS


