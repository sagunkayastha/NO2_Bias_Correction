import pandas as pd
import xarray as xr
import numpy as np
import os

from utils.filter_norm import prep_data  # , unnormalize_no2_vcd
from utils.train_func import epoch_train, epoch_eval
from torch.utils.data import DataLoader
import ray
import torch

import torch.optim as optim

from loss import amf_loss, IOA_Loss, Huber_loss

from amf_dataloader import TempoDataset
from args_model import Args
from models.fno_23 import AMFPredictor_FNO


# from scipy.stats import linregress
from utils.utils import calculate_metrics
import random

seed = 42
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
        self.combined_ds = self.combined_ds.where(
            self.combined_ds.Tempo_NO2_total <= 2e17, drop=True
        )

    def get_dataloaders(self):
        loaders_indices = {}

        # Get unique stations
        all_stations = np.unique(self.combined_ds.Station.values)

        for i, test_station in enumerate(all_stations):
            test_indices = np.where(self.combined_ds.Station.values == test_station)[0]

            train_indices = np.where(self.combined_ds.Station.values != test_station)[0]

            loaders_indices[str(test_station)] = [train_indices, test_indices]

        self.loader_indices = loaders_indices

    def save_values(self, values, norm_stats, dates=None):
        mmin = norm_stats["pandora_no2_smooth"]["min"]
        mmax = norm_stats["pandora_no2_smooth"]["max"]

        preds = values["preds"] * (mmax - mmin) + mmin
        targets = values["targets"] * (mmax - mmin) + mmin
        tempo = values["tempo"]
        dates = np.concat(dates)

        return [dates, preds, targets, tempo]

    def train_model(self, Station):
        train_indices, test_indices = (
            self.loader_indices[Station][0],
            self.loader_indices[Station][1],
        )

        train_ds = self.combined_ds.isel(time=train_indices)
        test_ds = self.combined_ds.isel(time=test_indices)

        train_ds, norm_stats = prep_data(train_ds, test=False)
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
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

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
        best_eval_loss = np.inf  # Start with a very low IOA

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

            eval_loss = total_eval_loss  # Since total_eval_loss = -IOA, invert it

            print(
                f"Train Loss: {abs(total_train_loss):.4f}, Eval loss: {eval_loss:.4f}"
            )

            # 🔹 Update Learning Rate Scheduler
            scheduler.step(eval_loss)

            # 🔹 Early Stopping Check
            if eval_loss < best_eval_loss:  # Lower loss is better
                best_eval_loss = eval_loss
                epochs_no_improve = 0
                # torch.save(
                #     model.state_dict(),
                #     os.path.join(self.args.run_folder, "models", "best_model.pth"),
                # )  # Save best model
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                eval_metrics, fig = calculate_metrics(
                    eval_values, norm_stats, plot=True, station=Station
                )
                fig.savefig(f"{args.run_folder}/plots/{Station}.png")
                print("Early stopping triggered.")
                values = self.save_values(eval_values, norm_stats, eval_dates)

                np.save(f"{args.run_folder}/values/{Station}.npy", values)
                return eval_metrics
                break
            if epoch == self.args.num_epochs - 1:
                eval_metrics, fig = calculate_metrics(
                    eval_values, norm_stats, plot=True, station=Station
                )
                fig.savefig(f"{args.run_folder}/plots/{Station}.png")
                values = self.save_values(eval_values, norm_stats, eval_dates)
                np.save(f"{args.run_folder}/values/{Station}.npy", values)

                print("Training complete.")
                return eval_metrics
                break

    def run(self):
        self.load_dataset()
        self.get_dataloaders()

        all_metrics = []
        for Station in np.unique(self.combined_ds.Station.values):
            print(f"Station: {Station}")
            model_folder = os.path.join(self.args.run_folder, "models")
            metrics_folder = os.path.join(self.args.run_folder, "metrics")
            plots_folder = os.path.join(self.args.run_folder, "plots")
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            if not os.path.exists(metrics_folder):
                os.makedirs(metrics_folder)
            if not os.path.exists(plots_folder):
                os.makedirs(plots_folder)

            eval_metrics = self.train_model(Station)
            all_metrics.append(eval_metrics)

            # break
        all_metrics = pd.DataFrame(all_metrics)
        all_metrics.to_csv(f"{self.args.run_folder}/metrics/all_metrics.csv")

    def run_ray(self):
        # Initialize Ray

        self.load_dataset()
        self.get_dataloaders()

        model_folder = os.path.join(self.args.run_folder, "models")
        metrics_folder = os.path.join(self.args.run_folder, "metrics")
        plots_folder = os.path.join(self.args.run_folder, "plots")
        values_folder = os.path.join(self.args.run_folder, "values")
        for folder in [model_folder, metrics_folder, plots_folder, values_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        @ray.remote(num_gpus=0.25)
        def train_model_remote(self, Station):  # Define Ray remote function
            try:
                return self.train_model(Station)
            except Exception as e:
                print(f"Error in {Station}: {e}")

        all_metrics = []
        stations = np.unique(self.combined_ds.Station.values)

        futures = [train_model_remote.remote(self, Station) for Station in stations]
        results = ray.get(futures)

        # futures = [
        #     train_model_remote.remote(self, fold) for fold in range(10)
        # ]  # Submit tasks to Ray
        # results = ray.get(futures)

        for eval_metrics in results:
            if eval_metrics is not None:
                all_metrics.append(eval_metrics)

        all_metrics = pd.DataFrame(all_metrics)
        all_metrics.to_csv(f"{self.args.run_folder}/metrics/all_metrics.csv")


if __name__ == "__main__":
    args = Args()

    run_folder = os.path.join(
        args.run_folder,
        "runs",
        args.run_name,
    )
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
    ray.init(address="auto")

    args.run_folder = run_folder

    model = Tempo_model(args)

    model.run_ray()
    ray.shutdown()
