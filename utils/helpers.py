import shutil

import numpy as np
import pandas as pd
from collections import Counter
import torch
import os
import yaml
import csv
from datetime import datetime
import config
import random


def load_labeled_paths(train_img_dir, train_labels_dir):
    df = pd.read_csv(train_labels_dir)
    df['label_index'] = df[['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']].idxmax(axis=1)
    label_map = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}
    df['label'] = df['label_index'].map(label_map)
    df['filepath'] = df['image'].apply(lambda x: os.path.join(train_img_dir, f"{x}.jpg"))
    return df['filepath'].tolist(), df['label'].tolist()


# def compute_class_weights(dataset, num_classes, device=None):
#     labels = dataset.label_paths  # already a list of integer class IDs
#     class_counts = Counter(labels)
#     total = len(labels)
#
#     weights = [total / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)]
#     print(weights)
#     return torch.tensor(weights, dtype=torch.float32, device=device)


def setup_run_folder(base_dir, run_config):
    """
    Create a timestamped run directory, save run config as YAML,
    and initialize a CSV logger.
    """
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M")
    run_dir = os.path.join(base_dir, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Convert torch.Tensor values to lists for YAML compatibility
    safe_config = {}
    for k, v in run_config.items():
        if isinstance(v, torch.Tensor):
            safe_config[k] = v.tolist()
        else:
            safe_config[k] = v

    # Save config to YAML
    yaml_path = os.path.join(run_dir, "config.yaml")
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(safe_config, f)

    # Prepare CSV log file
    csv_log_path = os.path.join(run_dir, f"log_{run_name}.csv")
    with open(csv_log_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "train_margin_loss",
            "train_recon_loss",
            "val_margin_loss",
            "val_recon_loss"
        ])

    # save model
    model_source_path = os.path.join(base_dir, "models", "model.py")
    model_target_path = os.path.join(run_dir, "model.py")
    if os.path.exists(model_source_path):
        shutil.copy(model_source_path, model_target_path)
    else:
        print("⚠️ model.py not found at expected path.")

    print(f"📁 Run directory created: {run_dir}")
    print(f"📄 model.py copied to:     {model_target_path}")
    print(f"📄 Config saved to:        {yaml_path}")
    print(f"📊 Log CSV initialized:    {csv_log_path}")

    return run_dir, csv_log_path


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32  # 2^32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_class_weights_from_labels(labels, num_classes):
    """
    Compute balanced class weights from a list of class labels.
    Assumes labels is a list of integers.
    Returns a normalized weight tensor (sum to 1).
    """
    counts = Counter(labels)
    total = sum(counts.values())

    weights = []
    for i in range(num_classes):
        freq = counts.get(i, 1) / total  # avoid division by zero
        weight = 1.0 / freq
        weights.append(weight)

    weights = torch.tensor(weights, dtype=torch.float32)
    return weights / weights.sum()  # normalize for stability


class Nadam(torch.optim.Optimizer):
    """Implements Nadam algorithm (a variant of Adam based on Nesterov momentum).

    It has been proposed in `Incorporating Nesterov Momentum into Adam`__.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        schedule_decay (float, optional): momentum schedule decay (default: 4e-3)

    __ http://cs229.stanford.edu/proj2015/054_report.pdf
    __ http://www.cs.toronto.edu/~fritz/absps/momentum.pdf
    """

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, schedule_decay=4e-3):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, schedule_decay=schedule_decay)
        super(Nadam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m_schedule'] = 1.
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                # Warming momentum schedule
                m_schedule = state['m_schedule']
                schedule_decay = group['schedule_decay']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                eps = group['eps']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                momentum_cache_t = beta1 * \
                                   (1. - 0.5 * (0.96 ** (state['step'] * schedule_decay)))
                momentum_cache_t_1 = beta1 * \
                                     (1. - 0.5 *
                                      (0.96 ** ((state['step'] + 1) * schedule_decay)))
                m_schedule_new = m_schedule * momentum_cache_t
                m_schedule_next = m_schedule * momentum_cache_t * momentum_cache_t_1
                state['m_schedule'] = m_schedule_new

                # Decay the first and second moment running average coefficient
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg_sq_prime = exp_avg_sq.div(1. - bias_correction2)

                denom = exp_avg_sq_prime.sqrt_().add_(group['eps'])

                p.data.addcdiv_(-group['lr'] * (1. - momentum_cache_t) / (1. - m_schedule_new), grad, denom)
                p.data.addcdiv_(-group['lr'] * momentum_cache_t_1 / (1. - m_schedule_next), exp_avg, denom)

        return loss
