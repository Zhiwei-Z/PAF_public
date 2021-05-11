## Environment setup
Create a virtual environment with command `conda create -n paf python=3.6`. After activating the virtual env with `conda activate paf`, install the required packages with `pip install -r requirements.txt`. At last run `pip install -e .`

## Running commands
Sample command for tuning experiment: 
`python launcher_scripts/execute.py -c configs/ant/base.json -t configs/ant/tuning.json -r 5 -p 2 -n testing_tuning`
or alternatively run the `.sh` files in this repo

## Script
The script `execute.py` uses a sweeper operator that performs a hyperparameter grid search specified by the tuning configuraion file passed to `tuning` whether through `--tuning` or `-t`. The flag `-c` or `--config` specifies a set of unchanged hyperparameters that is used for tuning, though it can be overwritten by tuning configuration files.

## Reproducing graphs
Running `execute.py` will create a directory `../paf-data` that store the experiment data. To reconstruct the graphs, run `plot.ipynb`.
