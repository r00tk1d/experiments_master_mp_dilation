# experiments_master_mp_dilation

## Setup ##
1. Clone this repo (`git clone https://github.com/r00tk1d/experiments_master_mp_dilation`)
2. Install conda (https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
3. Create conda env with python 3.10 (`conda create --name myenv python=3.10`)
4. Activate conda env (`conda activate myenv`)
5. Install (stumpy) dependencies (`conda install -c conda-forge -y numpy scipy numba matplotlib`)
6. Clone stumpy with dilation repo (`git clone https://github.com/r00tk1d/stumpy_master_mp_dilation`)
7. `cd stumpy_master_mp_dilation`
8. `pip install --editable .`
9. Install Jupyter Kernel (`conda install -n myenv ipykernel --update-deps --force-reinstall`)

## Good to Know ##
If you make changes in stumpy, restart the jupyter kernel to apply the changes and use them in the jupyter notebook

With `jupyter nbconvert --to python FILENAME.ipynb` you can create a python file out of the jupyter notebook to execute in the terminal. If you want to run the python file in the background: With `screen -Rd benchmark` create a new screen and run the created python file. To exit the screen `strg-a und strg-d`.

## Known Problems ##

Stumpy Import Error: version 'GLIBCXX_3.4.30' not found: `conda install -c conda-forge gcc=12.1.0`

Tests not Running:
- activate the correct conda environment

## Chains ##
Experiment Options:
- target_w or m
- anchored true or false
- groundtruth given or taken from no dilation run


## Segmentation ##
- unknown Change Points with fixed target range
- unknown Change Points with fixed window size
- known Change Points with fixed target range
- known Change Points with fixed window size
