#!/bin/bash

extract:
	./../ExtractDatabase/ExtractDatabase edge.txt 10000 10


run:
	python3 ReadIMDB.py IMDB.mtx edge.txt deg.txt

clean:
	- rm edge.txt deg.txt IMDB.txt
	rm -r [0-9]*/
