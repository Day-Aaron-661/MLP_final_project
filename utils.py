import numpy as np

def confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t], class_to_idx[p]] += 1
    return cm, classes


def accuracy(cm):
    return np.trace(cm) / np.sum(cm)


def precision(cm):
    num_classes = cm.shape[0]
    precisions = []
    for i in range(num_classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        if(TP + FP) > 0:
            precisions.append(TP / (TP + FP))
        else:
            precisions.append(0)
    return np.mean(precisions)


def recall(cm):
    num_classes = cm.shape[0]
    recalls = []
    for i in range(num_classes):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        if(TP + FN) > 0:
           recalls.append(TP / (TP + FN))
        else:
            recalls.append(0)
    return np.mean(recalls)


def f2_score(precision, recall):
    if precision + recall != 0:
        return (5 * (precision * recall)) / ((4 * precision) + recall)
    else:
        return 0