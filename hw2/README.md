
Light semantic parsing: SEMAFOR
I ran SEMAFOR, and tried to measure FScore on the frames of the hypothesis and reference sentences. This was pretty slow, and didn't seem to work very well.

Language Model: KenLM
I used the score from a language model via KenLM. I trained it using data I had from Penn Treebank, BNews, and the reference sentences of this task. I got modest improvements from this.

Sentiment Analysis: Text Blob
I tried running Sentiment Analysis from using the Text Blob package. The idea was that if the hypothesis sentiments disagreed with each other, we should take the one that matches the reference sentiment. This didn't seem to help very much.




There are three Python programs here (`-h` for usage):

 - `./evaluate` evaluates pairs of MT output hypotheses relative to a reference translation using counts of matched words
 - `./check` checks that the output file is correctly formatted
 - `./grade` computes the accuracy

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./evaluate | ./check | ./grade


The `data/` directory contains the following two files:

 - `data/train-test.hyp1-hyp2-ref` is a file containing tuples of two translation hypotheses and a human (gold standard) translation. The first 26208 tuples are training data. The remaining 24131 tuples are test data.

 - `data/train.gold` contains gold standard human judgements indicating whether the first hypothesis (hyp1) or the second hypothesis (hyp2) is better or equally good/bad for training data.

Until the deadline the scores shown on the leaderboard will be accuracy on the training set. After the deadline, scores on the blind test set will be revealed and used for final grading of the assignment.
