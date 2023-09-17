
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.optim as optim
import prettytable as pt

import numpy as np
import torch
import torch.optim as optim
import prettytable as pt
from numpy import random
from torch.backends import cudnn

from networksfinal import DeepSurv
from model import NegativeLogLikelihood
from datasets import SurvivalDataset
from utils import read_config
from utils import c_index
from utils import adjust_learning_rate
from utils import create_logger

def train(ini_file):
    ''' Performs training according to .ini file

    :param ini_file: (String) the path of .ini file
    :return best_c_index: the best c-index
    '''
    # reads configuration from .ini file
    config = read_config(ini_file)
    # builds network|criterion|optimizer based on configuration
    model = DeepSurv(config['network']).to(device)
    criterion = NegativeLogLikelihood(config['network']).to(device)

    optimizer = eval('optim.{}'.format(config['train']['optimizer']))(
        model.parameters(), lr=config['train']['learning_rate'])




    train_dataset = SurvivalDataset(config['train']['h5_file'], is_train=True)


    test_dataset = SurvivalDataset(config['train']['h5_file'], is_train=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_dataset.__len__())

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_dataset.__len__())


    best_c_index = 0
    flag = 0
    for epoch in range(1, config['train']['epochs']+1):##外循环控制循环轮次
        # adjusts learning rate
        lr = adjust_learning_rate(optimizer, epoch,
                                  config['train']['learning_rate'],
                                  config['train']['lr_decay_rate'])

        model.train()
        for X, y, e in train_loader:
            # makes predictions
            X = X.to(device)
            y = y.to(device)
            e = e.to(device)
            risk_pred = model(X).to(device)
            train_loss = criterion(risk_pred, y, e, model).to(device)#使用定义好的损失函数 `criterion`，计算当前 batch 的损失值 `train_loss`
            train_c = c_index(-risk_pred, y, e)
            # updates parameters
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
        # valid step
        model.eval()#将模型设置为评估模式 `eval()`，再使用验证数据集评估当前模型的性能
        for X, y, e in test_loader:#通过 `test_loader` 迭代读取测试集数据。真实值 `y`、事件发生状态 `e`
            # makes predictions
            X = X.to(device)
            y = y.to(device)
            e = e.to(device)
            with torch.no_grad():
                risk_pred = model(X).to(device)#得到风险预测值 `risk_pred`
                valid_loss = criterion(risk_pred, y, e, model).to(device)
                valid_c = c_index(-risk_pred, y, e)
                if best_c_index < valid_c:
                    best_c_index = valid_c
                    flag = 0
                    # saves the best model
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch}, os.path.join(models_dir, ini_file.split('\\')[-1]+'.pth'))
                else:
                    flag += 1
                    if flag >= patience:#这里的 `patience` 是一个超参数，用于控制模型训练的提前结束，
                       
                        return best_c_index

        print('\rEpoch: {}\tLoss: {:.8f}({:.8f})\tc-index: {:.8f}({:.8f})\tlr: {:g}'.format(
            epoch, train_loss.item(), valid_loss.item(), train_c, valid_c, lr), end='', flush=False)
    return best_c_index




if __name__ == '__main__':
    seed = 3045  
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True
    device = torch.device("cuda:3")
    # global settings
    logs_dir = 'logs'#定义了日志文件的目录。
    models_dir = os.path.join(logs_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    logger = create_logger(logs_dir)
    configs_dir =  # 定义了配置文件所在的目录。
    params = [
        ('Simulated Linear', 'linear.ini'),
        ('Simulated Nonlinear', 'gaussian.ini'),
        ('WHAS', 'whas.ini'),
        ('SUPPORT', 'support.ini'),
        ('METABRIC', 'metabric.ini'),
        ('Simulated Treatment', 'treatment.ini'),
        ('Rotterdam & GBSG', 'gbsg.ini')]
                              
    patience = 50 #提前停止次数
    # training
    headers = []
    values = []
    device = torch.device('cuda:3')


    for name, ini_file in params:
        logger.info('Running {}({})...'.format(name, ini_file))
        best_c_index = train(os.path.join(configs_dir, ini_file))
       
        headers.append(name)
        values.append('{:.6f}'.format(best_c_index))
       
        print('')
        logger.info("The best valid c-index: {}".format(best_c_index))
        logger.info('')
    # prints results
    
    