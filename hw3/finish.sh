#!/bin/sh

cat a.txt > output.txt
cat b.txt >> output.txt
cat c.txt >> output.txt
cat d.txt >> output.txt

cat output.txt | ./grade
