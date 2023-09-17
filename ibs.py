# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.optim as optim
import prettytable as pt
import numpy as np
from matplotlib import pyplot as plt

from networksfinal import DeepSurv
from networksfinal import NegativeLogLikelihood
from datasets import SurvivalDataset
from utils import read_config
from utils import c_index
from utils import adjust_learning_rate
from utils import create_logger
from scipy.stats import chi2, chi2_contingency
from sklearn.metrics import brier_score_loss

def integrated_brier_score(risk_pred, y, e, t_max=None):
    hr_pred = -risk_pred.squeeze()
    survival_prob = 1 / (1 + np.exp(hr_pred))
    squared_diffs = (survival_prob - e) ** 2

    #brier_score = torch.max(squared_diffs, axis=0).values.squeeze().numpy()
    brier_score = torch.mean(squared_diffs, axis=0).squeeze().numpy()

    #brier_score.repeat(y.shape[0], 1)
    #brier_score = brier_score_loss(e, survival_prob)
    if t_max is None:
        t_max = torch.max(y)
    y = y.squeeze().numpy()
    t_max = t_max.item()

    #brier_score = brier_score.item()  # 将标量张量转换为Python标量
    # 转换为NumPy数组


    ibs = np.trapz(brier_score, y) / t_max # 使用item()方法获取标量值
    print(ibs)

    return ibs

def chi_square_test(risk_pred, y, e, bins=10):
    hr_pred = -risk_pred.squeeze().cpu().numpy()
    e = e.squeeze().cpu().numpy()
    y = y.squeeze().cpu().numpy()
    observed_events = np.histogram(y[e == 1], bins=bins)[0]
    expected_events = np.histogram(hr_pred[e == 1], bins=bins)[0]
    result = chi2_contingency([observed_events, expected_events])
    chi2_stat = result[0]
    p_value = result[1]
    return chi2_stat, p_value


def mse(risk_pred, y, e):
    hr_pred = -risk_pred.squeeze()
    return np.mean((hr_pred - y) ** 2)
def plot_scatter(true_risk, predicted_risk, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(true_risk, predicted_risk, marker='o', color='blue', alpha=0.5)
    plt.xlabel('True Risk')
    plt.ylabel('Predicted Risk')
    plt.title(title)
    plt.grid(True)
    plt.show()

