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
