#!/bin/bash

extract:
	./../ExtractDatabase/ExtractDatabase edge.txt 36692 10

run:
	python3 ../dataClean.py -fi Email-Enron.txt

clean:
	- rm edge.txt outEdge* node-order*  nohup.out