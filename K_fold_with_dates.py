import os
import pickle
import random

import numpy as np
import pandas as pd
import xarray as xr

import ray
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from amf_dataloader import TempoDataset
from args_model import Args
from loss import Huber_loss
from models.fno_23 import AMFPredictor_FNO
from utils.filter_norm import prep_data  # , unnormalize_no2_vcd
from utils.train_func import epoch_eval, epoch_train
from utils.utils import calculate_metrics


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
        self.combined_ds = xr.open_dataset("combined_ds.nc").load()

        self.combined_ds = self.combined_ds.where(
            self.combined_ds.eff_cloud_fraction <= 0.2, drop=True
        )

    def get_dataloaders_k(self):
        loaders_indices = {}

        all_time_indices = np.arange(len(self.combined_ds.time))
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        for fold, (train_indices, test_indices) in enumerate(
            kf.split(all_time_indices)
        ):
            loaders_indices[fold] = [train_indices, test_indices]

        self.loader_indices = loaders_indices

    def get_dataloaders_g(self, n_splits=10):
        loaders_indices = {}

        # Extract unique station names
        unique_stations = np.unique(self.combined_ds.Station.values)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        station_folds = {
            fold: test_stations
            for fold, (_, test_stations) in enumerate(kf.split(unique_stations))
        }

        for fold, test_station_indices in station_folds.items():
            test_stations = unique_stations[test_station_indices]

            test_indices = np.where(
                np.isin(self.combined_ds.Station.values, test_stations)
            )[0]

            train_indices = np.where(
                ~np.isin(self.combined_ds.Station.values, test_stations)
            )[0]

            loaders_indices[fold] = [train_indices, test_indices]

        self.loader_indices = loaders_indices

        return loaders_indices

    def save_values(self, values, norm_stats, dates=None):
        mmin = norm_stats["pandora_no2_smooth"]["min"]
        mmax = norm_stats["pandora_no2_smooth"]["max"]

        preds = values["preds"] * (mmax - mmin) + mmin
        targets = values["targets"] * (mmax - mmin) + mmin
        tempo = values["tempo"]
        dates = np.concat(dates)

        return [dates, preds, targets, tempo]

    def train_model(self, fold):
        train_indices, test_indices = (
            self.loader_indices[fold][0],
            self.loader_indices[fold][1],
        )
        train_ds = self.combined_ds.isel(time=train_indices)
        test_ds = self.combined_ds.isel(time=test_indices)

        train_ds, norm_stats = prep_data(train_ds, test=False)

        with open(
            f"{self.args.run_folder}/norm/normalization_stats_{fold}.pkl", "wb"
        ) as f:
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
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        device = self.device

        model = AMFPredictor_FNO(
            input_dim_2d=len(self.args.vars_2d),
            input_dim_3d=len(self.args.vars_3d),
            hidden_dim=128,
            n_modes=12,
            fno_hidden=16,
            dropout_prob=0.3,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # criterion = IOA_Loss
        criterion = Huber_loss
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        patience = 10  # Number of epochs to wait for improvement
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
                total_eval_loss,
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

            eval_loss = total_eval_loss

            print(f"Train Loss: {abs(total_train_loss):.4f}, Eval IOA: {eval_loss:.4f}")

            scheduler.step(eval_loss)

            # Early Stopping Check
            if total_eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                epochs_no_improve = 0
                torch.save(
                    model.state_dict(),
                    os.path.join(self.args.run_folder, "models", f"{fold}.pth"),
                )
                torch.save(
                    model,
                    os.path.join(self.args.run_folder, "models", f"full_{fold}.pth"),
                )
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                eval_metrics, fig = calculate_metrics(
                    eval_values, norm_stats, plot=True, fold=fold
                )

                fig.savefig(f"{args.run_folder}/plots/{fold}.png")
                print("Early stopping triggered.")
                values = self.save_values(eval_values, norm_stats, eval_dates)
                np.save(f"{args.run_folder}/values/{fold}.npy", values)
                torch.save(
                    model,
                    os.path.join(self.args.run_folder, "models", f"full_{fold}.pth"),
                )
                return eval_metrics
                break

            if epoch == self.args.num_epochs - 1:
                eval_metrics, fig = calculate_metrics(
                    eval_values, norm_stats, plot=True, fold=fold
                )
                fig.savefig(f"{args.run_folder}/plots/{fold}.png")
                values = self.save_values(eval_values, norm_stats, eval_dates)
                np.save(f"{args.run_folder}/values/{fold}.npy", values)
                print("Training complete.")
                torch.save(
                    model,
                    os.path.join(self.args.run_folder, "models", f"full_{fold}.pth"),
                )
                return eval_metrics
                break

    def run_ray(self):
        self.load_dataset()

        if "G" in self.args.run_name:
            self.get_dataloaders_g()
        elif "K" in self.args.run_name:
            self.get_dataloaders_k()

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

        @ray.remote(num_gpus=0.25)
        def train_model_remote(self, fold):
            return self.train_model(fold)

        all_metrics = []
        futures = [train_model_remote.remote(self, fold) for fold in range(10)]
        results = ray.get(futures)  # Retrieve results

        for eval_metrics in results:
            all_metrics.append(eval_metrics)

        all_metrics = pd.DataFrame(all_metrics)
        all_metrics.to_csv(f"{self.args.run_folder}/metrics/all_metrics.csv")


if __name__ == "__main__":
    args = Args()

    ray.init(address="auto")
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
    ray.shutdown()
