#!/bin/bash

extract:
	./../ExtractDatabase/ExtractDatabase edge.txt 7115 10

run:
	python3 ../dataClean.py -fi Wiki-Vote.txt

clean:
	- rm edge.txt outEdge* node-order*  nohup.out
