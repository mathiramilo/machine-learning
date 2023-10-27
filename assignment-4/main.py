import matplotlib.pyplot as plt
import itertools
import torch
import torch.nn as nn
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import FeedForwardNN

import os


def show_image(tensor):
    image_transform = transforms.ToPILImage()
    image = tensor / 2 + 0.5  # Denormalizacion
    image = image_transform(image)
    image.show()


def train_epoch(
    model,
    train_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.CrossEntropyLoss,
):
    total_loss = 0
    runs = 0
    model.train(True)

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss
        runs += 1

    avg_loss = total_loss / runs

    print(f"Perdida promedio en entrenamiento: {avg_loss}")
    return avg_loss


def test_model(model, test_loader: DataLoader, loss_fn: nn.CrossEntropyLoss):
    model.train(False)
    total_loss = 0
    runs = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss
            runs += 1
            _, predicted = torch.max(outputs.data, 1)
            total += len(labels)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / runs
    avg_precision = correct / total

    print(f"Perdida promedio en testeo: {avg_loss}")
    print(f"Precision promedio en testeo: {avg_precision}")

    return avg_loss, avg_precision


def save_plot(prefix: str):
    postfix = 1
    while os.path.isfile(f"{prefix}{postfix}.png"):
        postfix += 1
    plt.savefig(f"{prefix}{postfix}.png")


def plot_loss_precision(
    train_loss_history: list,
    test_accuracy_history: list,
    save_fig: bool = False,
):
    """Grafica la perdida y la precision en funcion de las epocas.

    Args:
        train_loss_history (list): lista de arrays perdidas en el entrenamiento
        test_accuracy_history (list): lista de arrays precisiones en el testeo
    """

    plt.figure(figsize=(10, 12))
    plt.subplot(2, 1, 1)
    for i, marker in zip(range(len(train_loss_history)), itertools.cycle(['-', '--', ':'])):
        train_loss_history[i] = [x.detach().numpy()
                                 for x in train_loss_history[i]]
        plt.plot(train_loss_history[i],
                 label=f"M{i+1} Training Loss", linewidth=2, linestyle=marker)

    plt.legend(bbox_to_anchor=(1.25, 0.6), loc='center right')

    plt.title("Loss Over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    plt.subplot(2, 1, 2)
    for i, marker in zip(range(len(test_accuracy_history)), itertools.cycle(['-', '--', ':'])):
        plt.plot(test_accuracy_history[i],
                 label=f"M{i+1} Validation Accuracy", linewidth=2, linestyle=marker)

    plt.legend(bbox_to_anchor=(1.25, 0.6), loc='center right')

    plt.title("Accuracy Over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")

    if save_fig:
        save_plot(prefix="loss_precision")

    plt.show()


def plot_loss_comparation(
    train_loss_history, test_loss_history, title="Loss Over Iterations",
    save_fig: bool = False,
):
    """Grafica de comparacion de perdidas entre el entrenamiento y el testeo en funcion de las epocas."""

    train_loss_history_np = [x.detach().numpy() for x in train_loss_history]
    test_loss_history_np = [x.detach().numpy() for x in test_loss_history]

    plt.figure(figsize=(12, 4))
    plt.plot(train_loss_history_np, label="Training Loss")
    plt.plot(test_loss_history_np, label="Test Loss")
    plt.legend()
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    if save_fig:
        save_plot(prefix="loss_comparation")

    plt.show()


def run_model(model, optimizer, loss_fn, train_loader, test_loader, epochs=10):
    # Arrays para registrar la perdida y la precision
    train_loss_history = []
    test_loss_history = []
    test_accuracy_history = []

    # Entrenamiento y testeo
    for epoch in range(epochs):
        print(f"Epoca nr: {epoch}")

        avg_train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        avg_test_loss, avg_precision = test_model(model, test_loader, loss_fn)

        train_loss_history.append(avg_train_loss)
        test_loss_history.append(avg_test_loss)
        test_accuracy_history.append(avg_precision)

    return train_loss_history, test_loss_history, test_accuracy_history
