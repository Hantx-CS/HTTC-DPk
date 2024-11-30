#!/bin/bash

DataSets := IMDB
Epsilons := 0.5 1 2
Numbers := $(shell seq 10000 10000 100000)
version=2
epsilon=1
kAnony=2
Number = 0


.PHONY: all run clean single

merge:
	$(foreach e,$(Epsilons),\
		python3 -u merge.py -e $(e); \
		)

echos:
	$(foreach e,$(Epsilons),\
		make echo Number=$(e); \
		)

echo:
	echo "Number = $$Number";

test:
	$(foreach dataset,$(DataSets),\
		mkdir -p $(dataset); \
		)

run:
	$(foreach e,$(Epsilons),\
		make single epsilon=$(e); \
		)

single:
	mkdir -p IMDB-single-eps$(epsilon);
	$(foreach n,$(Numbers),\
		nohup python3 -u TriangleCount.py -d IMDB -v $(version) -e $(epsilon) -k $(kAnony) -n $(n) > IMDB-single-eps$(epsilon)/"n"$(n)"_nohup.out" 2>&1 & \
		)

# all:
# 	$(foreach e,$(Epsilons),\
# 		mkdir -p IMDB-eps$(e); \
# 		nohup python3 -u TriangleCount.py -d IMDB -v $(version) -e $(e) -k $(kAnony) > IMDB-eps$(e)/"nohup.out" 2>&1 & \
# 		)

clean:
	-rm nohup.out *.log *.dat
	-$(foreach e,$(Epsilons),\
		rm -r IMDB-single-eps$(e);)
