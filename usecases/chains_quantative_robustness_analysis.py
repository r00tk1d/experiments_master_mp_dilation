#!/usr/bin/env python
# coding: utf-8

# # Chain (directional trend) # 
# Every pair of consecutive subsequences in a chain must be connected by both a forward arrow and a backward arrow. The key component of drifting is that the time series must contain chains with clear directionality
# 
# Stumpy Tutorial Time Series Chains:
# https://stumpy.readthedocs.io/en/latest/Tutorial_Time_Series_Chains.html
# 
# Matrix Profile VII: Time Series Chains Calibration Instruction:
# https://docs.google.com/presentation/d/1-jEynFIkjDR88QFtbHN2Iz8DXY8wMVet/edit#slide=id.p1
# 
# Robust Time Series Chain Discovery with Incremental Nearest Neighbors:
# https://sites.google.com/view/robust-time-series-chain-22 

# In[1]:


import core.testdata as testdata
import core.utils as utils
import core.calculate as calculate
import core.visualize as visualize
import core.results as results

import pandas as pd


# ## Chain Robustness (Robustness Paper) ##
# 
# recall and precision: hits are with overlap > 50%

# In[2]:


# Parameter
use_case = "chains"
data_names = ['BME_1', 'BME_2', 'BME_3', 'BME_4', 'BME_5', 'CBF_1', 'CBF_2', 'CBF_3', 'CBF_4', 'CBF_5', 'ChlorineConcentration_1', 'ChlorineConcentration_2', 'ChlorineConcentration_3', 'ChlorineConcentration_4', 'ChlorineConcentration_5', 'ECG200_1', 'ECG200_2', 'ECG200_3', 'ECG200_4', 'ECG200_5', 'ECG5000_1', 'ECG5000_2', 'ECG5000_3', 'ECG5000_4', 'ECG5000_5', 'ECGFiveDays_1', 'ECGFiveDays_2', 'ECGFiveDays_3', 'ECGFiveDays_4', 'ECGFiveDays_5', 'FreezerRegularTrain_1', 'FreezerRegularTrain_2', 'FreezerRegularTrain_3', 'FreezerRegularTrain_4', 'FreezerRegularTrain_5', 'FreezerSmallTrain_1', 'FreezerSmallTrain_2', 'FreezerSmallTrain_3', 'FreezerSmallTrain_4', 'FreezerSmallTrain_5', 'Lightning7_1', 'Lightning7_2', 'Lightning7_3', 'Lightning7_4', 'Lightning7_5', 'Plane_1', 'Plane_2', 'Plane_3', 'Plane_4', 'Plane_5', 'SonyAIBORobotSurface1_1', 'SonyAIBORobotSurface1_2', 'SonyAIBORobotSurface1_3', 'SonyAIBORobotSurface1_4', 'SonyAIBORobotSurface1_5', 'SonyAIBORobotSurface2_1', 'SonyAIBORobotSurface2_2', 'SonyAIBORobotSurface2_3', 'SonyAIBORobotSurface2_4', 'SonyAIBORobotSurface2_5', 'Trace_1', 'Trace_2', 'Trace_3', 'Trace_4', 'Trace_5', 'TwoLeadECG_1', 'TwoLeadECG_2', 'TwoLeadECG_3', 'TwoLeadECG_4', 'TwoLeadECG_5', 'TwoPatterns_1', 'TwoPatterns_2', 'TwoPatterns_3', 'TwoPatterns_4', 'TwoPatterns_5', 'UMD_1', 'UMD_2', 'UMD_3', 'UMD_4', 'UMD_5', 'Wafer_1', 'Wafer_2', 'Wafer_3', 'Wafer_4', 'Wafer_5']


# In[3]:


max_dilation = 15

cols = ['dataname']
dilation_sizes = ["d"+str(d+1) for d in range(max_dilation)]
cols += dilation_sizes

recall_table = pd.DataFrame(columns=cols)
precision_table = pd.DataFrame(columns=cols)
f1_table = pd.DataFrame(columns=cols)

# Bulk Experiment
for count, data_name in enumerate(data_names):
    print(f'Starting Experiment {count+1}/{len(data_names)}: {data_name}')
    T = testdata.load_from_mat("../data/" + use_case + "/robustness/ts/" + data_name + ".mat", "ts")
    l = testdata.load_from_mat("../data/" + use_case + "/robustness/ts/" + data_name + ".mat", "l")

    ground_truth = testdata.load_gt("../data/" + use_case + "/robustness/gt/" + data_name + ".mat", "idx_tsc")

    # Hyperparameter
    target_w = int(l)
    m = None
    non_overlapping = False # if True, overlapping chains are filtered
    offset = False # if offset=True, the chains with dilation are calculated with a starting offset of the chain without dilation

    # calculate
    calculate.chains(T, max_dilation, data_name, use_case, ground_truth, offset, non_overlapping, target_w, m)

    # evaluate
    recalls, precisions, f1_scores = utils.get_metrics_for_experiment(max_dilation, data_name, use_case, offset, non_overlapping, target_w, m, ground_truth)
    recall_row = [data_name] + recalls
    recall_table.loc[len(recall_table)] = recall_row
    precision_row = [data_name] + precisions
    precision_table.loc[len(precision_table)] = precision_row
    f1_row = [data_name] + f1_scores
    f1_table.loc[len(f1_table)] = precision_row


    # visualize:
    print(f'Ground Truth Chain: {ground_truth}')
    visualize.chains(max_dilation, data_name, use_case, offset, non_overlapping, target_w, m, ground_truth, visualize_chains=False)
results.save_stats(recall_table, "../results/chains/recalls.csv")
results.save_stats(precision_table, "../results/chains/precisions.csv")
results.save_stats(f1_table, "../results/chains/f1_scores.csv")


# In[4]:


max_dilation = utils.calculate_max_d_from_m(275, 6000, max_d=100)
print(max_dilation)

