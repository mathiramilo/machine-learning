import pickle
import os
import torch
import torch.nn as nn
import gc


def save_model(model: nn.Module, path=f"models/model.pth"):
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, path=f"models/model.pth"):
    model.load_state_dict(torch.load(path))


def save_data(model_id: str, data_id, data, delete_after_save=True):
    if not os.path.exists(f"models/{model_id}"):
        os.makedirs(f"models/{model_id}")
    with open(f"models/{model_id}/{data_id}.pkl", "wb") as f:
        pickle.dump(data, f)
    if delete_after_save:
        del data
        data = None


def save_model_results(model_id: str, train_loss_history, test_loss_history, test_accuracy_history, delete_after_save=True):
    save_data(model_id, "train_loss_history",
              train_loss_history, delete_after_save)
    save_data(model_id, "test_loss_history",
              test_loss_history, delete_after_save)
    save_data(model_id, "test_accuracy_history",
              test_accuracy_history, delete_after_save)
    if delete_after_save:
        gc.collect()


def load_data(model_id: str, data_id):
    with open(f"models/{model_id}/{data_id}.pkl", "rb") as f:
        data = pickle.load(f)
    return data


def load_model_results(model_id: str, flags=(1, 1, 1)):
    train_loss_history, test_loss_history, test_accuracy_history = None, None, None
    if flags[0]:
        train_loss_history = load_data(model_id, "train_loss_history")
    if flags[1]:
        test_loss_history = load_data(model_id, "test_loss_history")
    if flags[2]:
        test_accuracy_history = load_data(model_id, "test_accuracy_history")
    return train_loss_history, test_loss_history, test_accuracy_history
