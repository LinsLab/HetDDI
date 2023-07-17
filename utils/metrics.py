from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score, roc_auc_score, accuracy_score
import numpy as np

def multi_class_eval(labels, pred, show=False):
    pred = [sample.argmax() for sample in pred]
    acc = accuracy_score(labels, pred)
    f1 = f1_score(labels, pred, average='macro')
    precision = precision_score(labels, pred, average='macro', zero_division=0)
    recall = recall_score(labels, pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(labels, pred)

    if show:
        acc_scores = []
        pred_by_class = [[] for i in range(86)]
        for i in range(len(labels)):
            pred_by_class[labels[i]].append(pred[i])
        for i in range(len(pred_by_class)):
            if len(pred_by_class)>0:
                acc_score = sum(np.array(pred_by_class[i]) == i)/len(pred_by_class[i])
                acc_scores.append(acc_score)

        return acc * 100, f1 * 100, precision * 100, recall * 100, kappa * 100, acc_scores
    else:
        return acc * 100, f1 * 100, precision * 100, recall * 100, kappa * 100

def binary_class_eval(labels, pred):
    auc = roc_auc_score(labels, pred)

    pred = np.round(pred)
    acc = accuracy_score(labels, pred)
    f1 = f1_score(labels, pred, zero_division=0)
    precision = precision_score(labels, pred, zero_division=0)
    recall = recall_score(labels, pred, zero_division=0)

    return acc * 100, f1 * 100, precision * 100, recall * 100, auc * 100

def multi_label_eval(labels, pred):
    pred = np.round(pred)
    acc = accuracy_score(labels, pred)
    f1 = f1_score(labels, pred, average='macro')
    precision = precision_score(labels, pred, average='macro')
    recall = recall_score(labels, pred, average='macro')
    auc = roc_auc_score(labels, pred)

    return acc * 100, f1 * 100, precision * 100, recall * 100, auc * 100
