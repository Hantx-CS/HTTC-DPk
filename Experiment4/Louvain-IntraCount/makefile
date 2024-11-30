#!/bin/bash

DataSets := Wiki-Vote Cit-HepTh Email-Enron IMDB
version=2
epsilon=2
kAnony=2


accuracy:
	$(foreach dataset,$(DataSets),\
		python3 -u accuracy.py -d $(dataset); \
		)

test:
	$(foreach dataset,$(DataSets),\
		mkdir -p $(dataset); \
		)

run:
	$(foreach dataset,$(DataSets),\
		mkdir -p $(dataset); \
		nohup python3 -u TriangleCount.py -d $(dataset) -v $(version) -k $(kAnony) > $(dataset)/"nohup.out" 2>&1 & \
		)

clean:
	-rm nohup.out *.log *.dat
	-$(foreach dataset,$(DataSets), \
		rm -r $(dataset);)
