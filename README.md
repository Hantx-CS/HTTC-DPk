# TDPTC

### Directory Structure
- data/			&emsp;datasets.
- Experiment1/		&emsp;Testing the performance on randomly partitioned data with varying privacy budgets.
- Experiment2/		&emsp;Testing the performance under different dataset sizes.
- Experiment3/	  &emsp;Testing the performance under different anonymity levels $k$.
- Experiment4/		&emsp;Testing the performance on Louvain and randomly partitioned data with a 1:9 ratio and varying privacy budgets.
- LICENSE.txt		&emsp;MPL license.
- README.md		&emsp;This file.

### Prepare for IMDB
Download the [IMDB dataset](https://www.cise.ufl.edu/research/sparse/matrices/Pajek/IMDB.html) and place the dataset in data/, and run the following commands:
```
  cd data/
  python3 ReadIMDB.py IMDB.mtx edges.csv deg.csv
  cd ../
```

### Prepare for Scale experiment
Generate 10 datasets of a specified scale(such as 10000) using the following command, which extracts a subset from the original dataset according to the defined scale. This allows for testing the performance of algorithms at different scales.The code for extracting 10 subsets with a scale of 10,000 from the IMDB dataset is shown below.
```
  cd data/create_cpp
  make
  ./CreateDatabase ../edges.csv 10000 10
  cd ../../
```

"The extracted datasets are saved in the same location as the original dataset by default. For instance, the 10 subsets generated by the above code will be stored under `/data`. After generation, please move the subsets to the required experimental directory, such as moving different scale datasets extracted from IMDB to the `experiment_Scale` directory."


### Run Experiments
Copy the dataset files from the `data/` directory to each `experiment_*` folder. Modify the `sourceFile` in the Makefile to match the name of the dataset and set `EPSILON` to the desired value. For the Fraction and Scale experiments, this value can be set in the Makefile. For other experiments, modify `EPSILON` directly in `TriangleCount.py`. Adjust the `SLEEPTIME` appropriately to prevent CPU overload during the 300 cycle experiments, with each time conducting 10 cycle experiments, based on the performance capabilities of your device.

For each `experiment_*` folder, the Python files can be executed using the commands `make run`, `make one`, or `make run_IMDB` specified in the Makefile. The default number of runs is set to 300, but this can be adjusted in the Makefile as needed.


Copy `convert.py` and `mean.py` into each `experiment_*` folder. Running `make mean` will calculate the average results from repeated experiments.
