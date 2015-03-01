#!/usr/bin/python
# coding=utf-8

import xml.etree.ElementTree as ET
import sys, string, codecs
from pprint import pprint

def printall(sentence_xml):
    SEARCH = [sentence_xml]
    while SEARCH:
        node = SEARCH.pop()
        print node.tag, node.attrib, 
        if node.text:
            print node.text.strip()
        else:
            print
        for child in node:
            SEARCH.append(child)

def getFrames(sentence_xml):
    frames = {}
    SEARCH = [sentence_xml]
    while SEARCH:
        node = SEARCH.pop(0)
        if node.tag == "annotationSet":
            key = node.attrib["frameName"]
            frames[key] = frames.get(key, 0) + 1
            continue
        for child in node:
            SEARCH.append(child)
    return frames

def getSentence(sentence_xml):
    SEARCH = [sentence_xml]
    while SEARCH:
        node = SEARCH.pop(0)
        if node.text.strip():
            return node.text
        for child in node:
            SEARCH.append(child)
    return None

def stripText(text):
    exclude = set(string.punctuation)
    stripped = ""
    for ch in text:
        if ch in exclude:
            ch = ""
        stripped += ch
    stripped = stripped.strip()
    stripped = " ".join(stripped.split())

    stripped = stripped.replace("quot","").replace("LRB","").replace("RRB","")
    
    #stripped = ''.join(ch for ch in text if ch not in exclude)
    #together = " ".join(stripped)
    return stripped

def word_matches(h, ref):
    return sum([1 for w in h if w in ref])

def precision_recall(h, ref):
    precision = 0.0
    recall = 0.0
    word_matches = {}

    if len(h) > 0:
        for word in h:
            if word in ref:
                precision += 1
        precision /= len(h)

        hset = set(h)
        for word in ref:
            if word in hset:
                recall += 1
        recall /= len(ref)

    return precision, recall



def frameRecall(h_frames, r_frames):
    N = sum(r_frames.values())
    num = 0.0
    for key in h_frames:
        current = h_frames[key]
        target = r_frames.get(key, 0)
        if current < target:
            num += current
        elif current > target:
            num += target
        else:
            num += current
    if N == 0:
        return 0.0
    return num / N

def framePrecision(h_frames, r_frames):
    N = sum(h_frames.values())
    num = 0.0
    for key in h_frames:
        current = h_frames[key]
        target = r_frames.get(key, 0)
        
        if current < target:
            num += current
        elif current > target:
            num += target
        else:
            num += current
    if N == 0:
        return 0.0
    return num / N

def evaluate(metric1, metric2, scores, answersfile):
    if metric1 > metric2:
        scores[-1] += 1
        answersfile.write("-1\n")
    elif metric1 < metric2:
        scores[1] += 1
        answersfile.write("1\n")
    elif metric1 == metric2:
        scores[0] += 1
        answersfile.write("0\n")

filename = sys.argv[1]

print "begin parsing XML"
tree = ET.parse(filename)
print "finished parsing XML"

root = tree.getroot()

print "begin storing sentences"
sentences = []
DFS = [root]
while DFS:
    node = DFS.pop()

    if node.tag == "sentence":
        sentences.append(node)
        continue

    for child in node:
        DFS.insert(0, child)

print "finished storing sentences"
print "(", len(sentences), "sentences)"

sentence_text = []
parsed_sentences = codecs.open("output.sentences.txt", "w", "utf8")

original_sentence_filename = sys.argv[2]
original_sentences = codecs.open(original_sentence_filename, "r", "utf8")

answers = open("results.txt", "w")

scores = {-1:0, 0:0, 1:0}
j = 0
for sentence1,sentence2,sentence3 in zip(sentences[0::3], sentences[1::3], sentences[2::3]):
    #printall(sentence1)
    text1 = stripText(getSentence(sentence1))
    h1_frames = getFrames(sentence1)
    parsed_sentences.write(text1+"\n")
    
    #printall(sentence2)
    text2 = stripText(getSentence(sentence2))
    h2_frames = getFrames(sentence2)
    parsed_sentences.write(text2+"\n")

    #printall(sentence3)
    text3 = stripText(getSentence(sentence3))
    r_frames = getFrames(sentence3)
    parsed_sentences.write(text3+"\n")

    """
    otext1 = stripText(original_sentences.next())
    otext2 = stripText(original_sentences.next())
    otext3 = stripText(original_sentences.next())
    """

    if len(r_frames) < 5:
        if len(text1) == 0 and len(text2) == 0 and len(text3) == 0:
            continue
        
        if len(text3.split()) > 10:
            p1,r1 = precision_recall(text1, text3)
            p2,r2 = precision_recall(text2, text3)

            LAMBDA = 0.9
            result1,result2 = (0.0, 0.0)
            if p1 + r1 > 0:
                result1 = r1*p1 / ((LAMBDA) * r1 + (1 - LAMBDA) * p1)
            if p2 + r2 > 0:
                result2 = r2*p2 / ((LAMBDA) * r2 + (1 - LAMBDA) * p2)
            evaluate(result1, result2, scores, answers)
        else:
            match1 = word_matches(text1, text3)
            match2 = word_matches(text2, text3)
            evaluate(match1, match2, scores, answers)

        continue

    LAMBDA = 0.5
    r1 = frameRecall(h1_frames, r_frames)
    p1 = framePrecision(h1_frames, r_frames)

    result1 = 0
    if r1 + p1 > 0:
        result1 = r1*p1 / ((LAMBDA) * r1 + (1 - LAMBDA) * p1)

    r2 = frameRecall(h2_frames, r_frames)
    p2 = framePrecision(h2_frames, r_frames)

    result2 = 0
    if r2 + p2 > 0:
        result2 = r2*p2 / ((LAMBDA) * r2 + (1 - LAMBDA) * p2)

    #sys.stderr.write(str(r1) + " " + str(r2))

    if result1 == result2:
        """
        p1,r1 = precision_recall(text1, text3)
        p2,r2 = precision_recall(text2, text3)

        LAMBDA = 0.9
        result1,result2 = (0.0, 0.0)
        if p1 + r1 > 0:
            result1 = r1*p1 / ((LAMBDA) * r1 + (1 - LAMBDA) * p1)
        if p2 + r2 > 0:
            result2 = r2*p2 / ((LAMBDA) * r2 + (1 - LAMBDA) * p2)
            """
        evaluate(result1, result2, scores, answers)
    else:
        evaluate(result1, result1, scores, answers)
    #j += 1

sys.stderr.write("SCORES: "+str(scores)+"\n")

values = sum(scores.values())
sys.stderr.write(str(values) + " SENTENCE TRIPLES\n")
sys.stderr.write(str(values*3) + " SENTENCES\n")

    

"""
output_filename = "original.output.sentences.txt"
output = codecs.open(output_filename, "w", "utf8")

missed_sentences_filename = "missed_sentences.txt"
output = codecs.open(missed_senteces_filename, "w", "utf8")
#pprint(missed_sentences)
for sentence,j in missed_sentences:
    output.write(str(j) + " " +sentence+"\n")
"""


