import argparse
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import torch

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def train(model, train_data, val_data, lr, epoch, weight=1, balance=True, path=None):
    best_loss, best_acc, best_F = 10, 0.0, 0.0
    print('train model')
    model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, weight])).cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(epoch):
        train_loss = 0.0
        model.train()
        predictions = np.empty((0, 2))
        label_all = []

        for idx, (text, label, _) in enumerate(tqdm(train_data)):
            train_x = torch.tensor(np.array([item.cpu().detach().numpy() for item in text])).cuda().T
            train_y = label.cuda()
            optimizer.zero_grad()
            pred = model(train_x)
            loss = criterion(pred, train_y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            # print([x.grad for x in optimizer.param_groups[0]['params']])  # 打印梯度
            predictions = np.vstack((predictions, pred.cpu().detach().numpy()))
            label_all += train_y.tolist()
        label_all = np.array(label_all)
        train_loss /= len(train_data)

        accuracy = np.sum(np.argmax(predictions, axis=1) == label_all) / len(label_all)
        _, _, f2, _ = precision_recall_fscore_support(label_all, np.argmax(predictions, axis=1), average='macro')
        print(
            f'epoch {epoch} train done! average train loss is {train_loss:.5f}, '
            f'average train accuracy is {accuracy:.5f}, average train F-measure is {f2:.5f}')

        val_loss, val_acc, val_F = test(model, val_data)
        if balance:
            if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
                best_acc = val_acc
                best_loss = val_loss
                print(
                    f'Save model! Best validation accuracy is {val_acc:.5f}, Best validation F-measure is {val_F:.5f}')
                model.save_pretrained(path)
        else:
            if val_F > best_F or (val_F == best_F and val_loss < best_loss):
                best_F = val_F
                best_loss = val_loss
                print(
                    f'Save model! Best validation accuracy is {val_acc:.5f}, Best validation F-measure is {val_F:.5f}')
                model.save_pretrained(path)


# 模型测试
def test(model, test_data, weight=1):
    print('test model')

    criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, weight])).cuda()  # 加权交叉熵损失

    model = model.cuda()
    model.eval()
    predictions = np.empty((0, 2))
    label_all = []
    test_loss = 0.0
    for idx, (text, label, _) in enumerate(tqdm(test_data)):
        train_x = torch.tensor(np.array([item.cpu().detach().numpy() for item in text])).cuda().T
        train_y = label.cuda()
        train_y = train_y.to(torch.int64)
        pred = model(train_x)
        loss = criterion(pred, train_y)
        test_loss += loss.item()
        predictions = np.vstack((predictions, pred.cpu().detach().numpy()))
        label_all += train_y.tolist()
    label_all = np.array(label_all)
    test_loss /= len(test_data)

    accuracy = np.sum(np.argmax(predictions, axis=1) == label_all) / len(label_all)

    P, R, F, _ = precision_recall_fscore_support(label_all, np.argmax(predictions, axis=1), average='macro')

    print(
        f'average test loss is {test_loss:.5f}, average test accuracy is {accuracy:.5f}, '
        f'average test Precision is {P:.5f}, average test Recall is {R:.5f}, average test F-measure is {F:.5f}')
    return test_loss, accuracy, F
