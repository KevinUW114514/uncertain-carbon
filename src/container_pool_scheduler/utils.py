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
import pandas as pd

SCHED_DIR = Path(__file__).resolve().parents[0]
sys.path.append(str(SCHED_DIR))

import huawei_data as data


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


def calc_percentile_stats(error_rates, overall_error_rate: float):
    print(f"max error rate: {np.argmax(error_rates)}: {np.max(error_rates)}")
    percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
    percentile_values = np.percentile(error_rates, percentiles)

    lines = []
    lines.append("=" * 80)
    lines.append(f"Overall mean error rate: {overall_error_rate:.2%}")
    lines.append("=" * 80)
    for p, val in zip(percentiles, percentile_values):
        lines.append(f"{p}th percentile error rate: {val:.2%}")

    log_text = "\n".join(lines)

    # Print to console
    print(log_text)

    # Save to file
    with open(f"inference_results.log", "w") as f:
        f.write(log_text + "\n")


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
    optimiser = optim.Adam(
        lr=learning_rate, params=model.parameters(), weight_decay=3e-4
    )
    dataloaders = data.get_dataloaders(datasets=datasets, train_batch_size=batch_size)

    loss_fn = F.mse_loss
    losses = {"train": [], "valid": []}

    # Wrap epochs in a progress iterator, but DON'T overwrite num_epochs
    if use_tqdm:
        from tqdm.auto import tqdm

        epoch_iter = tqdm(
            range(num_epochs), leave=True, disable=not use_tqdm, dynamic_ncols=True
        )
    else:
        epoch_iter = range(num_epochs)

    total_train = len(dataloaders["train"].dataset)
    x = dataloaders["train"].dataset[0]
    best_loss, best_idx = float("inf"), -1
    for epoch in epoch_iter:
        model.train()

        running_train_loss = 0.0
        samples_seen = 0

        for i, (x, y, _, _, _) in enumerate(dataloaders["train"]):
            x, y = x.to(device), y.to(device)
            # print(f"x: {x[0]}")
            # print(f"x[0].shape: {x[0].shape}")
            # print(f"x.shape: {x.shape}")
            # print(f"y: {y[0]}")
            # input("debug")
            # print(x.shape)
            # input("debug")
            out = model(x)
            optimiser.zero_grad()
            loss = loss_fn(out, y)
            loss.backward()
            optimiser.step()

            # track running average (nice stable metric per epoch)
            bs = x.size(0)
            # running_train_loss += loss.item() * bs
            running_train_loss += torch.abs(out - y).sum().item()
            samples_seen += bs

            # record step-wise loss if you want to keep your current log structure
            # step = i * batch_size + bs
            # losses["train"].append([epoch * total_train + step, loss.item()])

        # end of epoch: compute validation ONCE
        valid_loss = lstm_evaluate(device, model, dataloaders["valid"], samples_seen)[
            "loss"
        ]
        avg_train_loss = running_train_loss / max(1, samples_seen) * 100

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_idx = epoch
            print("=== New best model found! Saving... ===")
            print(f"=== Valid loss: {best_loss:.4f} at epoch {best_idx} ===")

        # store a point for the epoch (align step at end of epoch)
        # losses["valid"].append([epoch * total_train + samples_seen, valid_loss])

        # update UI ONCE per epoch
        if use_tqdm:
            epoch_iter.set_description(f"Epoch {epoch+1}/{num_epochs}")
            epoch_iter.set_postfix(
                train_loss=f"{avg_train_loss:.4f}%", valid_loss=f"{valid_loss:.4f}%"
            )
            # epoch_iter.refresh()  # optional
        else:
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"train loss={avg_train_loss:.4f} | valid loss={valid_loss:.4f}"
            )

    return model, losses


def lstm_evaluate(
    device: str, model: nn.Module, valid_loader: DataLoader, samples_seen: int
):
    loss_fn = F.mse_loss
    model = model.eval().to(device)
    for i, (x, y, _, _, _) in enumerate(valid_loader):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        loss = torch.abs(out - y).sum() / samples_seen * 100

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

    best_loss, best_idx = float("inf"), -1

    for epoch in epoch_iter:
        prediction_network.train()
        running_train_loss = 0.0
        samples_seen = 0

        # ---- training loop ----
        for i, (x, y, _, _, _) in enumerate(dataloaders["train"]):
            x, y = x.to(device), y.to(device)
            # print(y[:, 0, 1:])
            # input("debug")
            # print(x[0, :-3, 1:])
            # x = x[:, :-3, 1:]
            # x = x.reshape(x.size(0), -1)
            # print(x.shape)
            # print(x[:, :, 0].shape)
            # input("debug")
            out = prediction_network((x[:, :, 0].unsqueeze(-1), x[:, :-3, 1:].reshape(x.size(0), -1)))

            optimiser.zero_grad()
            loss = loss_fn(out, y[:, 0])
            loss.backward()
            optimiser.step()

            bs = x.size(0)
            # running_train_loss += loss.item() * bs
            running_train_loss += torch.abs(out - y[:, 0]).sum().item()
            samples_seen += bs

            # step = i * batch_size + bs
            # losses["train"].append([epoch * total_train + step, loss.item()])

        # ---- validation after each epoch ----
        valid_loss = evaluate_prediction_network(
            device, prediction_network, dataloaders["valid"], samples_seen
        )
        avg_train_loss = running_train_loss / max(1, samples_seen) * 100
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_idx = epoch
            print("=== New best model found! Saving... ===")
            print(f"=== Valid loss: {best_loss:.4f} at epoch {best_idx} ===")
        # losses["valid"].append([epoch * total_train + samples_seen, valid_loss])

        # ---- update tqdm ONCE per epoch ----
        if use_tqdm:
            epoch_iter.set_description(f"Epoch {epoch+1}/{num_epochs}")
            epoch_iter.set_postfix(
                train_loss=f"{avg_train_loss:.4f}%", valid_loss=f"{valid_loss:.4f}%"
            )
        else:
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"train loss={avg_train_loss:.4f} | valid loss={valid_loss:.4f}"
            )

    return prediction_network, losses


def evaluate_prediction_network(
    device: str, model: nn.Module, valid_loader: DataLoader, samples_seen: int
):
    loss_fn = F.mse_loss
    model = model.eval().to(device)
    for i, (x, y, _, _, _) in enumerate(valid_loader):
        break
    x, y = x.to(device), y.to(device)
    out = model((x[:, :, 0].unsqueeze(-1), x[:, :-3, 1:].reshape(x.size(0), -1)))
    # loss = loss_fn(out, y[:, :, 0])
    loss = torch.abs(out - y[:, 0]).sum() / samples_seen * 100

    return np.float32(loss.cpu().detach().numpy())


def dropout_on(m: nn.Module):
    if type(m) in [torch.nn.Dropout, vd.LSTM]:
        m.train()


def dropout_off(m: nn.Module):
    if type(m) in [torch.nn.Dropout, vd.LSTM]:
        m.eval()


def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = np.abs(y_pred) + np.abs(y_true)

    # Avoid division by zero
    mask = denominator == 0
    denominator[mask] = 1

    return 200 * np.mean(numerator / denominator)


def inference(
    datasets: dict, model: nn.Module, mc_dropout: bool = False, batch_size: int = 1
):

    device = get_device()
    valid_loader = data.get_dataloaders(datasets=datasets)["inference"]
    model.to(device)

    if mc_dropout:
        model = model.apply(dropout_on)
    else:
        model = model.apply(dropout_off)

    json_data = read_json_params("train_invocation_rate_normalization.json")
    train_mu, train_sigma = json_data["mu"], json_data["sigma"]

    for i, (x, y, x_hour, y_hour, x_start) in enumerate(valid_loader):
        x, y = x.to(device), y.to(device)
        res = []
        for _ in range(batch_size):
            res.extend(model((x, y[:, 0, 1:])).cpu().detach().tolist())
        mean = torch.mean(torch.tensor(res)).to(device)
        var = torch.var(torch.tensor(res))

        predicted = (x[:, 0, 0] + mean) * train_sigma + train_mu
        target = (
            ((y[:, 0, 0] + x[:, 0, 0]) * train_sigma + train_mu).to(device).squeeze(-1)
        )
        print(f"predicted_shape: {predicted.shape}, target_shape: {target.shape}")
        error_rates = torch.abs(predicted - target) / target
        # error_rate = (torch.abs(predicted - target) / target).sum() / len(x_hour) * 100
        error_rate = smape(target.cpu().numpy(), predicted.cpu().numpy())

        calc_percentile_stats(
            error_rates.cpu().numpy(),
            (torch.abs(predicted - target) / target).sum() / len(x_hour),
        )

        print(f"[inference] mean: {mean}, var: {var}, smape: {error_rate}")
        # print(f"[inference] predicted: {predicted}, target: {target}, error: {predicted - target}")

        # print(f"x_hour.shape: {x_hour.shape}, y_hour.shape: {y_hour.shape}")
        # print(f"predicted.shape: {predicted.shape}, target.shape: {target.shape}, error_rates.shape: {error_rates.shape}")

        df = pd.DataFrame(
            {
                "x_hour": x_hour,
                "x_start": x_start,
                "y_hour": y_hour,
                "predicted": predicted.detach().cpu().numpy(),
                "target": target.detach().cpu().numpy(),  # same name as your tensor
                "error_rates": error_rates.detach().cpu().numpy(),
            }
        )
        df.to_csv(f"inference_results.csv", index=False)

    return mean, var
    # return {"loss": np.float32(loss.cpu().detach().numpy())}


def save(model: nn.Module, name: str, path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
    model_path = Path(path) / "{}.pt".format(name)
    torch.save(model, model_path)
    print(f"PyTorch model saved at {model_path}")


def read_json_params(path):
    with open(path) as json_file:
        params = json.load(json_file)
    return params
