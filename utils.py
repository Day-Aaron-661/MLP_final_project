import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, ConfusionMatrixDisplay

def confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    cm = sk_confusion_matrix(y_true, y_pred, labels=classes)
    return cm, classes


def plot_confusion_matrix(cm, classes, title="Confusion Matrix"):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    disp.ax_.set_title(title)
    plt.show()


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
