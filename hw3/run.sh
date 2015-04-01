#!/bin/sh

./decode -s 8 -i data/a > a.txt &
./decode -s 8 -i data/b > b.txt &
./decode -s 8 -i data/c > c.txt &
./decode -s 8 -i data/d > d.txt &
