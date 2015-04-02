There are three Python programs here (`-h` for usage):

 - `./decode` a simple non-reordering (monotone) phrase-based decoder
 - `./grade` computes the model score of your output

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./decode | ./grade


The `data/` directory contains the input set to be decoded and the models

 - `data/input` is the input text

 - `data/lm` is the ARPA-format 3-gram language model

 - `data/tm` is the phrase translation model



I tried permuting the input string in different ways. For example, I permuted spans of length 2 to distances of 1 away, or spans of length 3 to distances of 2 away, etc.
I ran the baseline decoder on all of these permuted strings (and the original), and chose the best result. From this best result, I took the input string that created it, and started over, recursively permuting that input string again. I did this until I could not get an improvement.



