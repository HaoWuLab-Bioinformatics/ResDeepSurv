from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import h5py
import torch
import torch.optim as optim
import prettytable as pt
from scipy import stats

import numpy as np
import torch
import torch.optim as optim
import prettytable as pt
from numpy import random
from torch.backends import cudnn

from networksorigin import DeepSurv
from resattnnetworks import NegativeLogLikelihood
from datasets import SurvivalDataset
from utils import read_config
from utils import c_index
from utils import adjust_learning_rate
from utils import create_logger
from utils import bootstrap_c_index
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import scipy.stats as st
import numpy as np
import torch
import scipy.stats as st
import h5py
def bootstrap_metric(predictmodel, dataset, N):
    def sample_dataset(dataset, sample_idx):
        sampled_dataset = {}
        for key, value in dataset.items():
            if isinstance(value, np.ndarray):
                sampled_dataset[key] = value[sample_idx].to(device)
            elif isinstance(value, torch.Tensor):
                sampled_dataset[key] = value[sample_idx.clone()].to(device)
            else:
                sampled_dataset[key] = torch.tensor([value[i] for i in sample_idx]).to(device)
        return sampled_dataset

    metrics = []
    size = len(dataset['x'])

    for _ in range(N):
        resample_idx = np.random.choice(size, size=size, replace=True)
        samp_dataset = sample_dataset(dataset, resample_idx)

        pred_y = predictmodel(samp_dataset['x']).to(device)
        metric = c_index(samp_dataset['t'], pred_y, samp_dataset['e'])
        metrics.append(metric)

    mean = np.mean(metrics)
    conf_interval = st.t.interval(0.95, len(metrics) - 1, loc=mean, scale=st.sem(metrics))
    return {
        'mean': mean,
        'confidence_interval': conf_interval
    }
