import torch


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


def confusion_matrix(y_pred, y_labels, nb_classes):
    results = torch.argmax(y_pred, dim=1)
    conf_matrix = [[0] * nb_classes for _ in range(nb_classes)]
    for i in range(len(results)):
        conf_matrix[results[i]][y_labels[i]] += 1
    return torch.Tensor(conf_matrix)


def balanced_accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    exist_classes = torch.unique(yb)
    cls_acc = 0
    for cls in exist_classes:
        cls_preds = torch.where(preds == cls, 1, 0)
        cls_yb = torch.where(yb == cls, 1, 0)
        cls_acc += (cls_preds*cls_yb).sum()/torch.count_nonzero(cls_yb).sum()
    return cls_acc/exist_classes.size(0)