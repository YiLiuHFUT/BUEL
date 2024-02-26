from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score


def cal_metrics(y_true, y_predict, y_score):
    y = []
    true_0 = 0
    true_1 = 0
    for i in range(len(y_true)):
        if y_true[i] == 0 or y_true[i] == 0.0:
            true_0 += 1
            y.append([1, 0])
        else:
            true_1 += 1
            y.append([0, 1])
    acc = accuracy_score(y_true, y_predict)
    p2, r2, f2, _ = precision_recall_fscore_support(y_true, y_predict, average='macro')
    auc = roc_auc_score(y, y_score, average='macro')

    return acc, p2, r2, f2, auc
