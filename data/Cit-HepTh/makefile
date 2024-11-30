#!/bin/bash

extract:
	./../ExtractDatabase/ExtractDatabase edge.txt 27770 10

run:
	python3 ../dataClean.py -fi Cit-HepTh.txt

clean:
	- rm edge.txt outEdge* node-order*  nohup.out