# HTTC-DPk

### Directory Structure
- data/			&emsp;datasets.
- Experiment1/		&emsp;Testing the performance on randomly partitioned data with varying privacy budgets.
- Experiment2/		&emsp;Testing the performance under different dataset sizes.
- Experiment3/	  &emsp;Testing the performance under different anonymity levels $k$.
- Experiment4/		&emsp;Testing the performance on Louvain and randomly partitioned data with a 1:9 ratio and varying privacy budgets.
- LICENSE.txt		&emsp;MPL license.
- README.md		&emsp;This file.

For the code related to **CentralLap$_{\triangle}$** and **Local2Round$_{\triangle}$**, please refer to the following GitHub repository: [https://github.com/LDPGraphStatistics/LDPGraphStatistics](https://github.com/LDPGraphStatistics/LDPGraphStatistics).

### Prepare for IMDB
Download the [IMDB dataset](https://www.cise.ufl.edu/research/sparse/matrices/Pajek/IMDB.html) and place the dataset in data/, and run the following commands:
```
  cd data/IMDB/
  make run
  cd ../../
```


### Run Experiments
Copy the dataset files from the `data/` directory to each `experiment_*` folder. Modify the `sourceFile` in the Makefile to match the name of the dataset and set `EPSILON` to the desired value. For the Fraction and Scale experiments, this value can be set in the Makefile. For other experiments, modify `EPSILON` directly in `TriangleCount.py`. Adjust the `SLEEPTIME` appropriately to prevent CPU overload during the 300 cycle experiments, with each time conducting 10 cycle experiments, based on the performance capabilities of your device.

For each `experiment_*` folder, the Python files can be executed using the commands `make run`, `make one`, or `make run_IMDB` specified in the Makefile. The default number of runs is set to 300, but this can be adjusted in the Makefile as needed.


Copy `convert.py` and `mean.py` into each `experiment_*` folder. Running `make mean` will calculate the average results from repeated experiments.
