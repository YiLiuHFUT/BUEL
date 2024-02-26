import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import LogNormal
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import os
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import math
from cal_metrics import cal_metrics


# 设置随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


def prior_mu_compute(input):
    prior_list = []
    for a in zip(input):
        model_list = []
        for i in range(len(a[0])):
            prior_unc = - a[0][i, 0] * args.hyp_unc
            model_list.append(prior_unc)
        tmp = torch.Tensor(model_list)
        prior_list.append(tmp.unsqueeze(0))

    output = torch.cat(prior_list, 0)
    return output


class ModelWeightNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ModelWeightNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size, bias=True, dtype=float)
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True, dtype=float)
        self.linear3 = nn.Linear(hidden_size, output_size, bias=False, dtype=float)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        input = input[:, :, 0]
        # input = torch.flatten(input, 1)
        output_1 = self.linear1(input)
        h_1 = self.relu(output_1)
        output_2 = self.linear2(h_1)
        h_2 = self.relu(output_2)
        output = self.linear3(h_2)
        # output = self.relu(output_3)
        return output


def reparameterize_log_normal(mu, sigma):
    torch.manual_seed(42)

    sample = torch.zeros(mu.shape[0], mu.shape[1], 100)
    for i in range(100):
        epsilon = torch.randn_like(mu)
        z = mu + sigma * epsilon
        sample[:, :, i] = torch.exp(z)
    y = torch.mean(sample, dim=2).cuda()
    return y


# kl损失
def compute_kl_div(posterior, prior_mu, prior_sigma, sample_weight):
    # Compute the KL divergence between the posterior and prior distributions
    q_mu = posterior.loc
    q_sigma = posterior.scale

    kl_div = torch.log(prior_sigma / q_sigma) + (q_sigma ** 2 + (q_mu - prior_mu) ** 2) / (
            2 * prior_sigma ** 2) - 0.5

    kl_div = torch.mul(kl_div, sample_weight.reshape(len(sample_weight), 1))

    return kl_div.sum() / prior_mu.size(0)


def train(model, train_data, test_loader_adv=None, test_loader_ben=None, num_epoch=10, lr=0.001, path=None):
    best_loss = 10
    model.train()
    criterion = torch.nn.NLLLoss(reduction='none').cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.cuda()

    # 训练
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    for epoch in tqdm(range(num_epoch)):
        train_loss = 0.0
        model.train()
        predictions = np.empty((0, 2))
        label_all = []

        for idx, (uncer, pred, label, sample_type) in enumerate(train_loader):
            # 设置benign和adversarial之间的trade-off
            sample_weight = []
            ben = 1.0
            adv = args.lamda
            pos = args.gamma
            neg = 1.0
            for i in range(len(sample_type)):
                if sample_type[i] == 'ben' and label[i].item() == 0:
                    sample_weight.append(ben * neg)
                elif sample_type[i] == 'ben' and label[i].item() == 1:
                    sample_weight.append(ben * pos)
                elif sample_type[i] == 'adv' and label[i].item() == 0:
                    sample_weight.append(adv * neg)
                elif sample_type[i] == 'adv' and label[i].item() == 1:
                    sample_weight.append(adv * pos)
            sample_weight = torch.tensor(sample_weight, requires_grad=True).cuda()

            optimizer.zero_grad()
            output = model(uncer.double().cuda())

            prior_mu = prior_mu_compute(uncer).cuda()
            prior_sigma = torch.ones_like(uncer[:, :, 0]).cuda()
            prior_sigma.fill_(math.sqrt(args.sigma))

            posterior_sigma = torch.ones_like(uncer[:, :, 0]).cuda()
            posterior_sigma.fill_(math.sqrt(args.sigma))
            posterior_mu = output - (posterior_sigma ** 2) / 2
            posterior = LogNormal(posterior_mu, posterior_sigma)

            # Re-sampling
            w = reparameterize_log_normal(posterior_mu, posterior_sigma)
            # Normalizing
            w_norm = w / w.sum(dim=-1, keepdim=True)
            predict = torch.sum(pred.cuda() * w_norm.unsqueeze(2), dim=1)
            # Compute the KL divergence between the posterior and prior distributions
            kl_loss = compute_kl_div(posterior, prior_mu, prior_sigma, sample_weight)
            classify_loss = criterion(torch.log(predict), label.cuda())
            classify_loss_mean = torch.mul(classify_loss, sample_weight).mean()
            loss = kl_loss * 0.3 + classify_loss_mean

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # print([x.grad for x in optimizer.param_groups[0]['params']])  # 打印梯度
            predictions = np.vstack((predictions, predict.cpu().detach().numpy()))
            label_all += label.tolist()

        label_all = np.array(label_all)
        accuracy = np.sum(np.argmax(predictions, axis=1) == label_all) / len(label_all)
        _, _, f2, _ = precision_recall_fscore_support(label_all, np.argmax(predictions, axis=1), average='macro')
        print(
            f'epoch {epoch} train done! average train loss is {train_loss:.5f}, average train accuracy is {accuracy:.5f}, average train F-measure is {f2:.5f}')

        val_loss_adv, val_acc_adv, val_p_adv, val_f_adv = test_pro(model, test_loader_adv, MC=args.MC)
        val_loss_ben, val_acc_ben, val_p_ben, val_f_ben = test_pro(model, test_loader_ben, MC=args.MC)
        if best_loss > val_loss_adv + val_loss_ben:
            # 保存训练完成后的模型参数
            best_loss = val_loss_adv
            print(f'Save model! Best validation loss is {val_loss_adv:.5f}')
            torch.save(model.state_dict(), os.path.join(path, 'weight_model_pro.pt'))
        print('train_loss:', train_loss)



def test_pro(model, test_loader, MC=False, path=None):
    model.eval()
    test_loss = 0
    criterion = torch.nn.NLLLoss(reduction='none').cuda()  # 交叉熵损失
    predictions = np.empty((0, 2))
    label_all = []
    for idx, (uncer, pred, label, sample_type) in enumerate(test_loader):
        # 设置benign和adversarial之间的trade-off
        sample_weight = []
        ben = 1.0
        adv = args.lamda
        pos = args.gamma
        if balance:
            neg = 1.0
        else:
            neg = 0.5
        for i in range(len(sample_type)):
            if sample_type[i] == 'ben' and label[i].item() == 0:
                sample_weight.append(ben * neg)
            elif sample_type[i] == 'ben' and label[i].item() == 1:
                sample_weight.append(ben * pos)
            elif sample_type[i] == 'adv' and label[i].item() == 0:
                sample_weight.append(adv * neg)
            elif sample_type[i] == 'adv' and label[i].item() == 1:
                sample_weight.append(adv * pos)
        sample_weight = torch.tensor(sample_weight, requires_grad=True).cuda()

        output = model(uncer.cuda())

        if MC:
            posterior_sigma = torch.ones_like(uncer[:, :, 0]).cuda()
            posterior_sigma.fill_(math.sqrt(args.sigma))
            posterior_mu = output - (posterior_sigma ** 2) / 2
            # Re-sampling
            w = reparameterize_log_normal(posterior_mu, posterior_sigma)

        else:
            w = torch.exp(output)

        # Normalizing
        w_norm = w / w.sum(dim=-1, keepdim=True)

        predict = torch.sum(pred.cuda() * w_norm.unsqueeze(2), dim=1)
        classify_loss = criterion(torch.log(predict), label.cuda())
        classify_loss_mean = torch.mul(classify_loss, sample_weight).mean()
        test_loss += classify_loss_mean.item()

        predictions = np.vstack((predictions, predict.cpu().detach().numpy()))
        label_all += label.tolist()
    label_all = np.array(label_all)
    test_loss /= len(test_loader)

    acc, p2, r2, f2, auc = cal_metrics(label_all, np.argmax(predictions, axis=1), predictions)

    report_pro = classification_report(label_all, np.argmax(predictions, axis=1), digits=4, output_dict=True)
    df_pro = pd.DataFrame(report_pro).transpose()

    print(
        f'average test accuracy is {acc:.5f}, average test Precision is {p2:.5f}'
        f', average test Recall is {r2:.5f}, average test F-measure is {f2:.5f}, average test auc is {auc:.5f}')
    return test_loss, acc, p2, f2
