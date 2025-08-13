"""
train_weather.py
=================

This script demonstrates how one might fine‑tune the Hierarchical Reasoning
Model (HRM) on spatio‑temporal climate data such as ERA5.  HRM was
originally designed for puzzle‑style reasoning tasks, but its recurrent
architecture can be adapted for time‑series prediction by providing
sequences of tokens (in this case multivariate grid points) and
predicting future values.  The code below provides a skeleton for
loading ERA5 data, constructing an appropriate dataset, instantiating
the HRM model, and running a simple fine‑tuning loop.

**NOTE:** This script is intended as illustrative pseudocode.  You will
need to adapt it to your specific environment (e.g. install
``xarray``/``netCDF4`` and download ERA5 data), adjust paths to the
HRM source code, and tune hyperparameters.  It is assumed that you
have forked the HRM repository and that the ``models`` package is on
your ``PYTHONPATH`` so that ``from models.hrm.hrm_act_v1 import
HierarchicalReasoningModel`` works.

The dataset loader uses ``xarray`` to read NetCDF files exported
from the Copernicus Climate Data Store.  It stacks multiple ERA5
variables into a single tensor, standardises them, and builds
sliding‑window sequences.  The target for each input sequence is the
subsequent ``pred_horizon`` hours of data for a chosen target
variable.  You can modify the target generation logic for other
forecasting objectives (e.g. multi‑step forecasting of multiple
variables, classification of extreme events, etc.).
"""

import os
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# In practice you will need xarray to load ERA5 NetCDF/GRIB files.  We
# attempt to import it, but it's not required for illustrating the
# dataset structure.  Remove the ``try``/``except`` when xarray is
# installed.
try:
    import xarray as xr  # type: ignore
except ImportError:
    xr = None  # type: ignore


class ERA5WeatherDataset(Dataset):
    """Custom ``Dataset`` that prepares sliding‑window sequences from ERA5.

    Each element in the dataset consists of an input tensor of shape
    ``(seq_len, n_vars, height, width)`` and a target tensor of
    shape ``(pred_horizon, height, width)``.  The input contains
    ``n_vars`` variables (e.g. u10, v10, mwd, mwp, tp, etc.)
    standardised to zero mean and unit variance across the training set.
    The target contains the future values of a single variable (e.g.
    total precipitation ``tp``) ``pred_horizon`` hours ahead.
    """

    def __init__(
        self,
        file_paths: List[str],
        variables: List[str],
        target_variable: str,
        seq_len: int = 24,
        pred_horizon: int = 6,
        spatial_downsample: int = 4,
    ) -> None:
        super().__init__()
        assert target_variable in variables, "Target variable must be one of the input variables"

        if xr is None:
            raise ImportError(
                "xarray is required to load ERA5 data; please install it via `pip install xarray`"
            )

        self.variables = variables
        self.target_variable = target_variable
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon

        # Load and concatenate ERA5 files along the time dimension.  Each
        # file may correspond to a different period (e.g. one month of
        # hourly data).  Only the specified variables are loaded.
        datasets = []
        for fp in file_paths:
            ds = xr.open_dataset(fp, engine="netcdf4").sel(variable=variables)
            # Reduce temporal resolution to hourly if necessary
            datasets.append(ds)
        full_ds = xr.concat(datasets, dim="valid_time")

        # Optionally downsample spatially to reduce sequence length.  ERA5
        # provides variables on a ~31 km grid【567655082715995†L29-L36】, which may be too fine for HRM.
        if spatial_downsample > 1:
            full_ds = full_ds.coarsen(latitude=spatial_downsample, longitude=spatial_downsample, boundary="trim").mean()

        # Convert the xarray Dataset into a NumPy array of shape
        # (time, n_vars, height, width).  Reorder the dimensions
        # accordingly.
        data_array = full_ds.to_array().transpose("valid_time", "variable", "latitude", "longitude").values  # type: ignore

        # Standardise each variable to zero mean and unit variance along the
        # time dimension.  Keep statistics for later denormalisation if
        # required.
        self.means = data_array.mean(axis=(0, 2, 3), keepdims=True)
        self.stds = data_array.std(axis=(0, 2, 3), keepdims=True) + 1e-8
        data_array = (data_array - self.means) / self.stds

        self.data = torch.tensor(data_array, dtype=torch.float32)

        # Precompute the number of possible sequences given the chosen
        # ``seq_len`` and ``pred_horizon``.
        self.max_start = self.data.shape[0] - seq_len - pred_horizon + 1

        # Identify the index of the target variable in ``variables``
        self.target_index = variables.index(target_variable)

    def __len__(self) -> int:
        return self.max_start

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Slice the input sequence and the prediction target.  The
        # input is a tensor of shape (seq_len, n_vars, H, W).  The
        # target is of shape (pred_horizon, H, W) containing the
        # future values of the chosen target variable.
        input_seq = self.data[idx : idx + self.seq_len]
        target_seq = self.data[
            idx + self.seq_len : idx + self.seq_len + self.pred_horizon,
            self.target_index,
        ]  # (pred_horizon, H, W)
        return input_seq, target_seq


class WeatherHRM(nn.Module):
    """
    Thin wrapper around the HRM architecture to adapt it for
    spatio‑temporal forecasting.  The wrapper flattens the spatial
    dimensions of the input sequence into a single sequence of tokens
    and reshapes the output back to the original grid.  It assumes the
    underlying HRM model processes sequences of length
    ``seq_len * H * W`` and outputs an equal‑length sequence.
    """

    def __init__(self, hrm_model: nn.Module, n_vars: int, h: int, w: int, pred_horizon: int):
        super().__init__()
        self.hrm = hrm_model
        self.n_vars = n_vars
        self.h = h
        self.w = w
        self.pred_horizon = pred_horizon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, n_vars, H, W)

        Returns:
            out: Tensor of shape (batch, pred_horizon, H, W)
        """
        b, t, c, h, w = x.shape
        # Flatten spatial dims and variables into token sequence.  We
        # concatenate the variable dimension into the embedding by
        # reshaping: (batch, seq_len, n_vars, H, W) -> (batch,
        # seq_len * H * W, n_vars)
        x_flat = x.permute(0, 1, 3, 4, 2).contiguous().view(b, t * h * w, c)

        # Pass through HRM.  The model is expected to return a tuple
        # ``(carry, metrics, preds)`` according to the original
        # implementation.  We extract the predictions tensor.
        carry, metrics, preds = self.hrm(carry=None, batch={"inputs": x_flat}, return_keys=["preds"])  # type: ignore

        # ``preds`` has shape (batch, seq_len * H * W, embedding_dim).
        # We take only the last ``pred_horizon`` time steps and reshape
        # back to (batch, pred_horizon, H, W).  You may need to
        # project the embedding to a scalar via a linear layer.
        pred_tokens = preds[:, -self.pred_horizon * h * w :].view(b, self.pred_horizon, h, w, -1)
        # For regression we take the last channel as the output.  If your
        # HRM output head already produces a scalar per token, adjust
        # accordingly.
        out = pred_tokens[..., 0]
        return out


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)


def main() -> None:
    """Entry point for fine‑tuning HRM on ERA5 weather forecasting."""
    # Paths to NetCDF files containing ERA5 data for training and
    # validation.  Each file should contain the same variables.
    train_files = ["/path/to/era5_train.nc"]
    val_files = ["/path/to/era5_val.nc"]

    # Define which variables to load from ERA5.  You can include as
    # many as you like, provided they exist in the NetCDF files.
    variables = [
        "u10",  # 10m u‑component of wind
        "v10",  # 10m v‑component of wind
        "t2m",  # 2m temperature
        "sp",   # surface pressure
        "tp",   # total precipitation
        "mwd",  # mean wave direction
        "mwp",  # mean wave period
    ]
    target_variable = "tp"  # predict future precipitation

    # Hyperparameters
    seq_len = 24          # number of past hours to use as input
    pred_horizon = 6      # number of hours to forecast
    batch_size = 4
    epochs = 10
    learning_rate = 1e-4
    spatial_downsample = 4  # reduce spatial resolution to speed up training

    # Prepare datasets and dataloaders
    train_ds = ERA5WeatherDataset(
        train_files,
        variables,
        target_variable,
        seq_len=seq_len,
        pred_horizon=pred_horizon,
        spatial_downsample=spatial_downsample,
    )
    val_ds = ERA5WeatherDataset(
        val_files,
        variables,
        target_variable,
        seq_len=seq_len,
        pred_horizon=pred_horizon,
        spatial_downsample=spatial_downsample,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

    # Instantiate the HRM model.  You must import the correct HRM
    # class from your fork of the HRM repository.  Adjust the import
    # statement according to your package structure.  The model
    # configuration (number of layers, hidden sizes, etc.) should be
    # chosen to balance accuracy and computational cost.
    from models.hrm.hrm_act_v1 import HierarchicalReasoningModel  # type: ignore

    # Example HRM configuration.  Consult the original HRM paper or
    # repository for recommended parameter values.  ``n_vocab`` is set
    # equal to the number of variables; ``input_dim`` can be taken
    # directly from the number of variables if you choose not to
    # project the inputs.  ``num_heads`` and ``hidden_dim`` control
    # the complexity of the model.
    hrm_config: Dict[str, int] = {
        "n_vocab": len(variables),
        "input_dim": len(variables),
        "hidden_dim": 128,
        "num_layers": 2,
        "num_heads": 4,
        "max_seq_len": seq_len * (train_ds.data.shape[2] * train_ds.data.shape[3]),
    }
    hrm_model = HierarchicalReasoningModel(**hrm_config)

    # Wrap the HRM to handle spatio‑temporal input/output shapes
    _, n_vars, height, width = train_ds.data.shape[1:]
    model = WeatherHRM(hrm_model, n_vars, height, width, pred_horizon)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimiser and loss.  For precipitation forecasting we use
    # Mean Squared Error.  You may choose other losses depending on
    # your application (e.g. MAE, quantile loss).
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

    # Save the fine‑tuned model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/hrm_weather_finetuned.pt")


if __name__ == "__main__":
    main()
