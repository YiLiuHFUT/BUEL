import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


def enable_dropout(model):
    """
    :param model:
    :return:
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def fix_batch_normalization(model):
    """
    :param model:
    :return:
    """
    for m in model.modules():
        if m.__class__.__name__.find('BatchNorm') != -1:
            m.eval()


def get_var_pro(model, test_data, time):
    enable_dropout(model.model)
    fix_batch_normalization(model.model)

    model_predictions = np.empty((time, 0))
    for idx, (text, label) in enumerate(tqdm(test_data)):
        train_x = torch.tensor(np.array([item.cpu().detach().numpy() for item in text])).cuda().T
        pred_batch = np.zeros((time, 2 * len(train_x)))
        for j in range(time):
            pred = model.model(train_x)
            a = F.softmax(torch.tensor(pred), dim=-1)
            pred_batch[j] = a.cpu().detach().numpy().reshape(1, 2 * len(train_x))
        model_predictions = np.hstack((model_predictions, pred_batch))

    model_var = np.var(model_predictions, axis=0)
    model_pro = model_predictions.mean(axis=0)
    model_pro = model_pro.reshape(len(test_data.dataset.data_all), 2)  # 转置
    return model_var, model_pro
