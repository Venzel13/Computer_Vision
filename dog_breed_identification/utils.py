import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn

from preproc import class_num
from trainer import Trainer


def plot_confusion_matrix(true, pred, encoder, classes):
    true = encoder.inverse_transform(true[0].cpu())
    pred = encoder.inverse_transform(pred[0].cpu())
    matrix = confusion_matrix(true, pred, labels=classes)

    plt.figure(figsize=(8, 8))
    sns.heatmap(
        matrix,
        yticklabels=classes,
        xticklabels=classes,
        annot=True,
        cmap="YlGnBu",
        cbar=False,
    )


nets = {}
nets["mobile"] = Trainer(
    model_name="mobilenet_v2",
    loss_fn=nn.CrossEntropyLoss(),
    optim_name="Adam",
    optim_param={"lr": 1e-5},
    n_classes=class_num,
    verbose=True,
)
nets["resnet"] = Trainer(
    model_name="resnet101",
    loss_fn=nn.CrossEntropyLoss(),
    optim_name="Adam",
    optim_param={"lr": 1e-5},
    n_classes=class_num,
    verbose=True,
)
nets["mobile-sgd"] = Trainer(
    model_name="mobilenet_v2",
    loss_fn=nn.CrossEntropyLoss(),
    optim_name="SGD",
    optim_param={"lr": 1e-5, "momentum": 0.9, "nesterov": True, "weight_decay": 1e-4},
    n_classes=class_num,
    verbose=True,
)
nets["resnet-sgd"] = Trainer(
    model_name="resnet101",
    loss_fn=nn.CrossEntropyLoss(),
    optim_name="SGD",
    optim_param={"lr": 1e-5, "momentum": 0.9, "nesterov": True, "weight_decay": 1e-4},
    n_classes=class_num,
    verbose=True,
)
