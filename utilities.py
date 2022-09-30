import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def moveTo(obj, device):
    if isinstance(obj, list):
        return [moveTo(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(moveTo(list(obj), device))
    elif isinstance(obj, set):
        return set(moveTo(list(obj), device))
    elif isinstance(obj, dict):
        to_ret = dict()
        for key, value in obj.items():
            to_ret[moveTo(key, device)] = moveTo(value, device)
        return to_ret
    elif hasattr(obj, "to"):
        return obj.to(device)
    else:
        return obj


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        print(f"Directory for model storage created at : {dirpath}")
    else:
        print(f"Directory for model storage exists at : {dirpath}")


def make_train_state(args):
    return {
        'stop_early': False,
        'early_stopping_step': 0,
        'early_stopping_best_val': 1e8,
        'learning_rate': args.learning_rate,
        'epoch_index': 0,
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': -1,
        'test_acc': -1,
        'model_filename': args.model_state_file
    }


def update_train_state(args, model, train_state):
    """Handle the training state updates.

    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better

    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """
    # save one model
    if train_state['epoch_index'] == 0:
        torch.save(obj=model.state_dict(), f=train_state['model_filename'])
        train_state['stop_early'] = False

    elif train_state['epoch_index'] >= 1:
        # loss at time 't-minus-1' , loss at time 't'
        loss_tm1, loss_t = train_state['val_loss'][-2:]

        # if loss worsened
        if loss_t >= train_state['early_stopping_best_val']:
            # update step
            train_state['early_stopping_step'] += 1
        # loss decreased
        else:
            # save the model
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(obj=model.state_dict(), f=train_state['model_filename'])
            # reset early stopping step
            train_state['early_stopping_step'] = 0
            train_state['early_stopping_best_val'] = loss_t

        # Stop early?
        train_state['stop_early'] = train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state


def save_train_state(train_state: dict, args):
    with open(args.train_state_file, 'w') as fp:
        json.dump(train_state, fp)


def load_train_state(args):
    with open(args.train_state_file, 'r') as fp:
        return json.load(fp)


def compute_accuracy(y_pred, y_target):
    y_target = y_target.cpu()
    y_pred_indices = (torch.sigmoid(y_pred) > 0.5).cpu().long()  # .max(dim=1)[1]
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def train_state_to_dataframe(train_state, columns):
    frame = pd.DataFrame()
    for column in columns:
        frame[column] = train_state[column]

    return frame
