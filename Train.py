import os
import pickle
import random

import numpy as np
import pandas as pd
import xarray as xr

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from amf_dataloader import TempoDataset
from args_model import Args
from models.fno_23 import AMFPredictor_FNO
from utils.filter_norm import prep_data  # , unnormalize_no2_vcd
from utils.train_func import epoch_eval, epoch_train
from utils.utils import calculate_metrics
from loss import Huber_loss
from sklearn.model_selection import train_test_split

seed = 43
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Tempo_model:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_dataset(self):
        # lat_lon_df = pd.read_csv(self.args.station_file)

        # all_ds = []
        # for i in range(len(lat_lon_df)):

        #     ds = xr.open_dataset(lat_lon_df.loc[i,'processed_file'])
        #     ds['Tempo_NO2_total'] = ds['vertical_column_troposphere'] + ds['vertical_column_stratosphere']
        #     ds = ds.assign_coords(Station=lat_lon_df.loc[i, "Station_number"])

        #     all_ds.append(ds)
        # combined_ds = xr.concat(all_ds, dim='time')
        # combined_ds = combined_ds.dropna(dim='time', how='any')
        # combined_ds = combined_ds.where(combined_ds.Tempo_NO2_total > 0, drop=True)
        # combined_ds = combined_ds.where(combined_ds.fitted_slant_column > 0, drop=True)

        self.combined_ds = xr.open_dataset("combined_ds.nc").load()

        self.combined_ds = self.combined_ds.where(
            self.combined_ds.eff_cloud_fraction <= 0.2, drop=True
        )

    def save_values(self, values, norm_stats):
        mmin = norm_stats["pandora_no2_smooth"]["min"]
        mmax = norm_stats["pandora_no2_smooth"]["max"]

        preds = values["preds"] * (mmax - mmin) + mmin
        targets = values["targets"] * (mmax - mmin) + mmin
        tempo = values["tempo"]

        return [preds, targets, tempo]

    def train_model(self):
        all_time_indices = np.arange(len(self.combined_ds.time))
        train_indices, test_indices = train_test_split(
            all_time_indices, test_size=0.2, shuffle=True, random_state=42
        )

        train_ds = self.combined_ds.isel(time=train_indices)
        test_ds = self.combined_ds.isel(time=test_indices)

        train_ds = self.combined_ds.isel()
        test_ds = self.combined_ds

        train_ds, norm_stats = prep_data(train_ds, test=False)

        with open(f"{self.args.run_folder}/norm/normalization_stats.pkl", "wb") as f:
            pickle.dump(norm_stats, f)

        test_ds, norm_stats = prep_data(test_ds, test=True, norm_stats=norm_stats)

        train_dataset = TempoDataset(
            train_ds,
            self.args.vars_2d,
            self.args.vars_3d,
            self.args.scd_var,
            self.args.vcd_target,
        )
        test_dataset = TempoDataset(
            test_ds,
            self.args.vars_2d,
            self.args.vars_3d,
            self.args.scd_var,
            self.args.vcd_target,
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.args.batch_size, shuffle=False
        )

        device = self.device

        model = AMFPredictor_FNO(
            input_dim_2d=len(self.args.vars_2d),
            input_dim_3d=len(self.args.vars_3d),
            hidden_dim=128,
            n_modes=12,
            fno_hidden=16,
            dropout_prob=0.3,
        ).to(device)

        # model = AMFPredictor_NoTransformer(
        #     input_dim_2d=len(self.args.vars_2d),
        #     input_dim_3d=len(self.args.vars_3d),
        # ).to(device)
        # model = AMFPredictor_NoFNO(
        #     input_dim_2d=len(self.args.vars_2d),
        #     input_dim_3d=len(self.args.vars_3d),
        # ).to(device)

        # model = AMFPredictor_res(
        #         input_dim_2d=len(self.args.vars_2d),
        #         input_dim_3d=len(self.args.vars_3d),
        #         num_layers_3d=3,
        #         hidden_dim=16,
        #     ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # criterion = IOA_Loss
        criterion = Huber_loss
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        patience = 10
        best_eval_loss = np.inf
        epochs_no_improve = 0

        for epoch in range(self.args.num_epochs):
            print(f"Epoch {epoch + 1}")
            (
                model,
                total_train_loss,
                physics_train_loss,
                vcd_train_loss,
                train_values,
                train_amf,
                # eval_dates,
            ) = epoch_train(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                lambda_physics=self.args.lambda_physics,
            )
            (
                eval_loss,
                physics_train_loss,
                vcd_train_loss,
                eval_values,
                eval_amf,
                eval_dates,
            ) = epoch_eval(
                model,
                test_loader,
                criterion,
                device,
                lambda_physics=self.args.lambda_physics,
            )

            # Since total_eval_loss = -IOA, invert it

            print(
                f"Train Loss: {abs(total_train_loss):.4f}, Eval Loss: {eval_loss:.4f}"
            )

            scheduler.step(eval_loss)

            # Early Stopping Check
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                epochs_no_improve = 0

                torch.save(
                    model,
                    os.path.join(self.args.run_folder, "models", "running_model.pth"),
                )
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                eval_metrics, fig = calculate_metrics(
                    eval_values, norm_stats, plot=True, fold=0
                )

                print("Early stopping triggered.")

                torch.save(
                    model,
                    os.path.join(self.args.run_folder, "models", "Best_model.pth"),
                )
                return eval_metrics
                break

            if epoch == self.args.num_epochs - 1:
                eval_metrics, fig = calculate_metrics(
                    eval_values, norm_stats, plot=True, fold=0
                )
                f
                print("Training complete.")
                torch.save(
                    model,
                    os.path.join(self.args.run_folder, "models", "Best_model.pth"),
                )
                return eval_metrics
                break

    def run(self):
        # Initialize Ray

        self.load_dataset()

        model_folder = os.path.join(self.args.run_folder, "models")
        metrics_folder = os.path.join(self.args.run_folder, "metrics")
        plots_folder = os.path.join(self.args.run_folder, "plots")
        values_folder = os.path.join(self.args.run_folder, "values")
        norm_folder = os.path.join(self.args.run_folder, "norm")

        for folder in [
            model_folder,
            metrics_folder,
            plots_folder,
            values_folder,
            norm_folder,
        ]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        metrics = self.train_model()

        metrics = pd.DataFrame(metrics)
        metrics.to_csv(f"{self.args.run_folder}/metrics/metrics.csv")


if __name__ == "__main__":
    args = Args()
    run_folder = os.path.join(
        args.run_folder,
        "runs",
        args.run_name,
    )
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)

    args.run_folder = run_folder

    model = Tempo_model(args)
    # model.run()
    model.run_ray()
