# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import pylab
import torch
import torch.optim as optim
import prettytable as pt

import numpy as np
import torch
import torch.optim as optim
import prettytable as pt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib import pyplot as plt
from numpy import random
from torch.backends import cudnn

from featurenet import DeepSurv
from resattnnetworks import NegativeLogLikelihood
from datasets import SurvivalDataset
from utils import read_config
from utils import c_index
from utils import adjust_learning_rate
from utils import create_logger
import h5py
from torch.utils.data import DataLoader

# 读取 HDF5 文件

localtime   = time.localtime()

TIMESTRING  = time.strftime("%m%d%Y%M", localtime)

def calculate_recs_and_antirecs(rec_trt, true_trt, dataset, print_metrics=True):
    if isinstance(true_trt, int):
        true_trt = dataset['x'][:,true_trt]

    # trt_values = zip([0,1],np.sort(np.unique(true_trt)))
    trt_values = enumerate(np.sort(np.unique(true_trt)))
    equal_trt = [np.logical_and(rec_trt == rec_value, true_trt == true_value) for (rec_value, true_value) in trt_values]
    rec_idx = np.array(np.logical_or(*equal_trt))
    rec_idx = np.array(rec_idx, dtype=bool)
    # original Logic
    # rec_idx = np.logical_or(np.logical_and(rec_trt == 1,true_trt == 1),
    #               np.logical_and(rec_trt == 0,true_trt == 0))

    rec_t = dataset['t'][rec_idx]
    antirec_t = dataset['t'][~rec_idx]
    rec_e = dataset['e'][rec_idx]
    antirec_e = dataset['e'][~rec_idx]

    if print_metrics:
        print("Printing treatment recommendation metrics")
        metrics = {
            'rec_median' : np.median(rec_t),
            'antirec_median' : np.median(antirec_t)
        }
        print("Recommendation metrics:", metrics)

    return {
        'rec_t' : rec_t,
        'rec_e' : rec_e,
        'antirec_t' : antirec_t,
        'antirec_e' : antirec_e
    }


def plot_survival_curves(rec_t, rec_e, antirec_t, antirec_e, experiment_name='', output_file=None):
    # Set-up plots
    plt.figure(figsize=(12, 3))
    ax = plt.subplot(111)

    # Fit survival curves
    kmf = KaplanMeierFitter()
    kmf.fit(rec_t, event_observed=rec_e, label=' '.join([experiment_name, "Recommendation"]))
    kmf.plot(ax=ax, linestyle="-")
    kmf.fit(antirec_t, event_observed=antirec_e, label=' '.join([experiment_name, "Without Recommendation"]))
    kmf.plot(ax=ax, linestyle="--")

    # Format graph
    plt.ylim(0, 1);
    ax.set_xlabel('Timeline (months)', fontsize='large')
    ax.set_ylabel('Percentage of Population Alive', fontsize='large')

    # Calculate p-value

    if output_file:
        plt.tight_layout()
        pylab.savefig(output_file)

def save_treatment_rec_visualizations(model, dataset, output_dir, trt_i=1, trt_j=0, trt_idx=0):
    trt_values = np.unique(dataset['x'][:, trt_idx])

    print("Recommending treatments:", trt_values)
    rec_trt = model.recommend_treatment(dataset['x'], trt_i, trt_j, trt_idx)
    rec_trt = torch.squeeze((rec_trt < 0).type(torch.int32))

    rec_dict = calculate_recs_and_antirecs(rec_trt, true_trt=dataset['x'][:, trt_idx], dataset=dataset)

    output_file = os.path.join(output_dir, '_'.join(['deepsurv', TIMESTRING, 'rec_surv.pdf']))
    print(output_file)
    plot_survival_curves(experiment_name='ResDeepSurv', output_file=output_file, **rec_dict)