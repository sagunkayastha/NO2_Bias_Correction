import torch
from torch.utils.data import Dataset
import numpy as np
import logging


class TempoDataset(Dataset):
    def __init__(self, ds, variables_2d, variables_3d, scd_var, target_var):
        """
        ds: xarray Dataset (normalized)
        variables_2d: List of 2D variable names
        variables_3d: List of 3D variable names
        scd_var: Slant column NO2 variable (SCD) used in AMF calculation
        target_var: Pandora NO2 VCD target variable for loss calculation
        """
        self.variables_2d = variables_2d
        self.variables_3d = variables_3d
        self.scd_var = scd_var
        self.target_var = target_var
        self.ds = ds

        # Extract 2D features
        self.x_2d = np.stack(
            [ds[var].values for var in variables_2d], axis=-1
        )  # (samples, num_2d_features)

        # Extract and transpose 3D features
        self.x_3d = np.stack(
            [ds[var].values for var in variables_3d], axis=1
        )  # (time, swt_level, features)
        self.x_3d = np.transpose(self.x_3d, (2, 1, 0))  # (batch, features, swt_level)

        self.x_3d_raw = np.stack(
            [ds[f"{var.split('_norm')[0]}_orig"].values for var in variables_3d], axis=1
        )
        self.x_3d_raw = np.transpose(self.x_3d_raw, (2, 1, 0))

        # Store SCD and Target VCD
        self.scd = ds[scd_var].values.reshape(-1, 1)  # Fitted slant column
        self.y = ds[target_var].values.reshape(-1, 1)  # Pandora NO2 VCD
        self.tempo_column = ds["Tempo_NO2_total"].values
        self.tempo_amf = ds["amf_total"].values
        self.dates = ds["time"].values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # logging.debug(f"TempoDataset __getitem__ idx: {idx}, inputs_2d shape: {inputs_2d.shape}, vcd shape: {vcd.shape}")
        return (
            torch.tensor(self.x_2d[idx], dtype=torch.float32),
            torch.tensor(self.x_3d[idx], dtype=torch.float32),
            torch.tensor(self.scd[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.x_3d_raw[idx], dtype=torch.float32),
            torch.tensor(self.tempo_column[idx], dtype=torch.float32),
            torch.tensor(self.tempo_amf[idx], dtype=torch.float32),
            torch.tensor(self.dates[idx], dtype=torch.float32),
        )


class TempoDataset_eval(Dataset):
    def __init__(self, ds, variables_2d, variables_3d, scd_var):
        """
        ds: xarray Dataset (normalized)
        variables_2d: List of 2D variable names
        variables_3d: List of 3D variable names
        scd_var: Slant column NO2 variable (SCD) used in AMF calculation
        target_var: Pandora NO2 VCD target variable for loss calculation
        """
        self.variables_2d = variables_2d
        self.variables_3d = variables_3d
        self.scd_var = scd_var

        self.ds = ds

        # Extract 2D features
        self.x_2d = np.stack(
            [ds[var].values for var in variables_2d], axis=-1
        )  # (samples, num_2d_features)

        # Extract and transpose 3D features
        self.x_3d = np.stack(
            [ds[var].values for var in variables_3d], axis=1
        )  # (time, swt_level, features)
        self.x_3d = np.transpose(self.x_3d, (2, 1, 0))  # (batch, features, swt_level)

        self.x_3d_raw = np.stack(
            [ds[f"{var.split('_norm')[0]}_orig"].values for var in variables_3d], axis=1
        )
        self.x_3d_raw = np.transpose(self.x_3d_raw, (2, 1, 0))

        # Store SCD and Target VCD
        self.scd = ds[scd_var].values.reshape(-1, 1)  # Fitted slant column
        # Pandora NO2 VCD
        self.tempo_column = ds["Tempo_NO2_total"].values
        self.tempo_amf = ds["amf_total"].values

    def __len__(self):
        return len(self.scd)

    def __getitem__(self, idx):
        # logging.debug(f"TempoDataset __getitem__ idx: {idx}, inputs_2d shape: {inputs_2d.shape}, vcd shape: {vcd.shape}")
        return (
            torch.tensor(self.x_2d[idx], dtype=torch.float32),
            torch.tensor(self.x_3d[idx], dtype=torch.float32),
            torch.tensor(self.scd[idx], dtype=torch.float32),
            torch.tensor(self.x_3d_raw[idx], dtype=torch.float32),
            torch.tensor(self.tempo_column[idx], dtype=torch.float32),
            torch.tensor(self.tempo_amf[idx], dtype=torch.float32),
        )
