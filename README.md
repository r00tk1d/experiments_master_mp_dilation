# Experiments Masterthesis: A Dilated Matrix Profile
This is the supporting code used for evaluating the dilated matrix profile. The experiments are conducted for the two application domains time series chain discovery and semantic segmentation.

## Directory structure ##   
This is the structure of the experiments with its important files for the thesis:
- data                          (time series data)
    - chains
        - robustness            (TSC22 Quantitative Analysis Dataset)
            - ...
        - chain_test_ .txt      (TSC17 Quantitative Analysis Dataset)
        - humangaittreadmill.mat (Case Study Qualitative Analysis)  
        - penguinshort.mat  (Case Study Qualitative Analysis)
        - tilttable.txt  (Case Study Qualitative Analysis)
        - webquery.txt  (Case Study Qualitative Analysis)           
    - (segmentation data is from TSSB package)
- experiments                   (code for the experiments in Jupyter Notebooks)
    - core                      (python code used in the Jupyter Notebooks)   
        - ...                   
    - chains_ .ipynb             (chain experiments)
    - segmentation_ .ipynb       (segmentation experiments)
- results
    - chains
        - _DATANAME_
            - _EXPERIMENTNAME_
                - \_plot_*.png (result overview plots)
                - _DATANAME_DILATIONSIZE_WINDOWSETTING_ (result data and visuals for specific parameters)
    - segmentation
        - segmentation_covering_ .csv (results from the experiments)
        - \_segmentation_covering_ .csv (condensed results from the experiments)
        - competitor_evaluation_ .ipynb (evaluates the condensed results from the experiments and compares them)
        - correlations (evaluates different correlations)
            - correlation_data_characteristics.ipynb (shows the correlation between specific data characteristics and dilation sizes)
            - correlation_global.ipynb (shows the correlation between the Covering score of fluss1/flussOracle and data characteristics)
- tests (tests for `stumpy_dil()`)
    - test_AA.py (tests for Self-joins)
    - test_AB.py (tests for AB-joins)
- unused files (old files with no meaning for the final thesis)

## Setup ##
1. Clone this repo: `git clone https://github.com/r00tk1d/experiments_master_mp_dilation`
2. [Download](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and install conda
3. Create conda env with python 3.10: `conda create --name myenv python=3.10`
4. Activate conda env: `conda activate myenv`
5. Install (stumpy) dependencies: `conda install -c conda-forge -y numpy scipy numba matplotlib`
6. Clone stumpy with dilation repo: `git clone https://github.com/r00tk1d/stumpy_master_mp_dilation`
7. `cd stumpy_master_mp_dilation`
8. `pip install --editable .`
9. Install Jupyter Kernel: `conda install -n myenv ipykernel --update-deps --force-reinstall`
10. [Download](https://sites.google.com/view/robust-time-series-chain-22) the TSC22 robustness dataset .mat files into the folders "data/chains/robustness/ts" and "data/chains/robustness/gt"


## Run the Jupyter Notebooks and apply changes ##
If you make changes in the imported code, restart the jupyter kernel to apply the changes and use them in the jupyter notebook.

With `jupyter nbconvert --to python FILENAME.ipynb` you can create a python file out of the jupyter notebook to execute in the terminal. If you want to run the python file in the background: With `screen -Rd benchmark` create a new screen and run the created python file. To exit the screen `strg-a und strg-d`.

## Error Fixes ##

Stumpy Import Error: version 'GLIBCXX_3.4.30' not found: `conda install -c conda-forge gcc=12.1.0`

