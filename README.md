# HTTC-DPk

### Directory Structure
- data/			&emsp;datasets.
- Experiment1/		&emsp;Testing the performance on randomly partitioned data with varying privacy budgets.
- Experiment2/		&emsp;Testing the performance under different dataset sizes.
- Experiment3/	  &emsp;Testing the performance under different anonymity levels $k$.
- Experiment4/		&emsp;Testing the performance on Louvain and randomly partitioned data with a 1:9 ratio and varying privacy budgets.
- LICENSE.txt		&emsp;MPL license.
- README.md		&emsp;This file.

For the code related to CentralLap△ and Local2Round△, please refer to the following GitHub repository: [https://github.com/LDPGraphStatistics/LDPGraphStatistics](https://github.com/LDPGraphStatistics/LDPGraphStatistics).

### Prepare for IMDB
Download the [IMDB dataset](https://www.cise.ufl.edu/research/sparse/matrices/Pajek/IMDB.html) and place the dataset in data/, and run the following commands:
```
  cd data/IMDB/
  make run
  cd ../../
  cd data/
  make dataClean
  cd ../
```


### Run Experiments
In each method folder within the **Experiment** directory, you can run `make run`. For example, in the **Experiment1/HTTC-DPk** folder, simply execute the following code:
```
  cd Experiment1/HTTC-DPk
  make run
  cd ../../
```

##### Prepare IMDB with different sizes for Experiment2

Download the necessary library files as specified in the **/data/ExtractDatabase/include/README.md** file, and then execute the following code:
```
  cd /data/ExtractDatabase
  make
  cd ../../
```

Next, modify the commands under the **extract** section in the **/data/IMDB/makefile** as needed, and then execute the following code. (The current code is capable of extracting 10 sets of subgraphs, each with 10,000 nodes.)
```
  cd /data/IMDB/
  make extract
  cd ../../
```

