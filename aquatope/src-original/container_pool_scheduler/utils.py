import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.encoder_decoder_dropout import *
from torch.utils.data import DataLoader

SCHED_DIR = Path(__file__).resolve().parents[0]
sys.path.append(str(SCHED_DIR))

import data


def get_device() -> str:
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    return torch.device(device)


from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train_encoder_decoder(
    device: str,
    model: nn.Module,
    datasets: dict,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    use_tqdm: bool = False,
) -> Tuple[nn.Module, dict]:
    model.to(device)
    optimiser = optim.Adam(lr=learning_rate, params=model.parameters())
    dataloaders = data.get_dataloaders(datasets=datasets, train_batch_size=batch_size)

    loss_fn = F.mse_loss
    losses = {"train": [], "valid": []}

    # Wrap epochs in a progress iterator, but DON'T overwrite num_epochs
    if use_tqdm:
        from tqdm.auto import tqdm
        epoch_iter = tqdm(range(num_epochs), leave=True, disable=not use_tqdm)
    else:
        epoch_iter = range(num_epochs)

    total_train = len(dataloaders["train"].dataset)

    for epoch in epoch_iter:
        model.train()

        running_train_loss = 0.0
        samples_seen = 0

        for i, (x, y) in enumerate(dataloaders["train"]):
            x, y = x.to(device), y.to(device)
            out = model(x)

            optimiser.zero_grad()
            loss = loss_fn(out, y)
            loss.backward()
            optimiser.step()

            # track running average (nice stable metric per epoch)
            bs = x.size(0)
            running_train_loss += loss.item() * bs
            samples_seen += bs

            # record step-wise loss if you want to keep your current log structure
            step = i * batch_size + bs
            losses["train"].append([epoch * total_train + step, loss.item()])

        # end of epoch: compute validation ONCE
        valid_loss = lstm_evaluate(device, model, dataloaders["valid"])["loss"]
        avg_train_loss = running_train_loss / max(1, samples_seen)

        # store a point for the epoch (align step at end of epoch)
        losses["valid"].append([epoch * total_train + samples_seen, valid_loss])

        # update UI ONCE per epoch
        if use_tqdm:
            epoch_iter.set_description(f"Epoch {epoch+1}/{num_epochs}")
            epoch_iter.set_postfix(train_loss=f"{avg_train_loss:.4f}",
                                   valid_loss=f"{valid_loss:.4f}")
            # epoch_iter.refresh()  # optional
        else:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"train loss={avg_train_loss:.4f} | valid loss={valid_loss:.4f}")

    return model, losses



def lstm_evaluate(device: str, model: nn.Module, valid_loader: DataLoader):
    loss_fn = F.mse_loss
    model = model.eval().to(device)
    for i, (x, y) in enumerate(valid_loader):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)

    return {"loss": np.float32(loss.cpu().detach().numpy())}


from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train_prediction_network(
    device: str,
    datasets: dict,
    prediction_network: nn.Module,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    use_tqdm: bool = True,
) -> Tuple[nn.Module, dict]:

    dataloaders = data.get_dataloaders(datasets=datasets, train_batch_size=batch_size)
    prediction_network.to(device)

    optimiser = optim.Adam(
        lr=learning_rate, params=prediction_network.model.parameters()
    )
    loss_fn = F.mse_loss
    losses = {"train": [], "valid": []}

    if use_tqdm:
        from tqdm.auto import tqdm
        epoch_iter = tqdm(range(num_epochs), leave=True)
    else:
        epoch_iter = range(num_epochs)

    total_train = len(dataloaders["train"].dataset)

    for epoch in epoch_iter:
        prediction_network.train()
        running_train_loss = 0.0
        samples_seen = 0

        # ---- training loop ----
        for i, (x, y) in enumerate(dataloaders["train"]):
            x, y = x.to(device), y.to(device)
            out = prediction_network((x, y[:, 0, 1:]))

            optimiser.zero_grad()
            loss = loss_fn(out, y[:, :, 0])
            loss.backward()
            optimiser.step()

            bs = x.size(0)
            running_train_loss += loss.item() * bs
            samples_seen += bs

            step = i * batch_size + bs
            losses["train"].append(
                [epoch * total_train + step, loss.item()]
            )

        # ---- validation after each epoch ----
        valid_loss = evaluate_prediction_network(
            device, prediction_network, dataloaders["valid"]
        )
        avg_train_loss = running_train_loss / max(1, samples_seen)
        losses["valid"].append(
            [epoch * total_train + samples_seen, valid_loss]
        )

        # ---- update tqdm ONCE per epoch ----
        if use_tqdm:
            epoch_iter.set_description(f"Epoch {epoch+1}/{num_epochs}")
            epoch_iter.set_postfix(train_loss=f"{avg_train_loss:.4f}",
                                   valid_loss=f"{valid_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"train loss={avg_train_loss:.4f} | valid loss={valid_loss:.4f}")

    return prediction_network, losses



def evaluate_prediction_network(
    device: str, model: nn.Module, valid_loader: DataLoader
):
    loss_fn = F.mse_loss
    model = model.eval().to(device)
    for i, (x, y) in enumerate(valid_loader):
        break
    x, y = x.to(device), y.to(device)
    out = model((x, y[:, 0, 1:]))
    loss = loss_fn(out, y[:, :, 0])

    return np.float32(loss.cpu().detach().numpy())


def save(model: nn.Module, name: str, path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
    model_path = Path(path) / "{}.pt".format(name)
    torch.save(model, model_path)
    print(f"PyTorch model saved at {model_path}")


def read_json_params(path):
    with open(path) as json_file:
        params = json.load(json_file)
    return params
