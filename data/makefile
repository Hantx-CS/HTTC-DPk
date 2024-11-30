#!/bin/bash

create:
	make -C create_cpp all

run:
	make -C Cit-HepTh run
	make -C Email-Enron run
	make -C Wiki-Vote run

dataClean:
	nohup python3 -u dataClean.py -fi IMDB/IMDB.txt &

clean:
	- rm nohup.out
	- make -C Cit-HepTh clean
	- make -C Email-Enron clean
	- make -C Wiki-Vote clean
	- make -C IMDB clean
