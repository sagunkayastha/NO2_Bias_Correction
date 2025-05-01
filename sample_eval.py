import xarray as xr
import torch
import numpy as np
import matplotlib.pyplot as plt
from amf_dataloader import TempoDataset
from args_model import Args
from models.fno_23 import AMFPredictor_FNO
from utils.filter_norm import prep_data

from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from matplotlib import colors
from scipy.stats import linregress


import random

seed = 43
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def IOA_Loss_np(o, p):
    ioa = 1 - (np.sum((o - p) ** 2)) / (
        np.sum((np.abs(p - np.mean(o)) + np.abs(o - np.mean(o))) ** 2)
    )
    return ioa


def plot_scatter_with_fit(
    ax, x, y, xlabel, ylabel, title, fig=None, metrics=None, var=None
):
    mab = metrics[f"mab_{var}"]
    IOA = metrics[f"ioa_{var}"]

    ax.set_aspect("equal", "box")  # Makes each subplot square

    hb = ax.hexbin(x, y, gridsize=200, cmap="turbo", norm=colors.LogNorm(), mincnt=1)

    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Count")
    cb.ax.tick_params(length=6, width=2)

    ax.tick_params(axis="both")

    # max_value = max(x) or max(y)
    max_value = max(x) or max(y)
    ax.set_xlim([0, max_value])
    ax.set_ylim([0, max_value])
    ax.plot([0, max_value], [0, max_value], linestyle="--", color="grey")

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    ax.plot(x, slope * x + intercept, color="red")
    ax.text(0.05, 0.95, f"R = {r_value:.2f}", transform=ax.transAxes, va="top")
    ax.text(0.05, 0.90, f"IOA = {IOA:.2f}", transform=ax.transAxes, va="top")
    num_data_points = len(x)
    ax.text(0.05, 0.80, f"slope = {slope:.2f}", transform=ax.transAxes, va="top")

    ax.text(
        0.05,
        0.85,
        f"MAB = {mab:.2f} × 10^15 molecules/cm²",
        transform=ax.transAxes,
        va="top",
    )
    ax.text(0.05, 0.75, f"n = {num_data_points:,}", transform=ax.transAxes, va="top")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def eval_model(test_ds, model_path=None, norm_path=None):
    model = torch.load(model_path, weights_only=False)

    test_ds, norm_stats = prep_data(test_ds, test=True, D2=True, stats_file=norm_path)
    test_dataset = TempoDataset(
        test_ds, args.vars_2d, args.vars_3d, args.scd_var, args.vcd_target
    )
    batch_size = 128
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    all_preds = []
    all_tempo = []
    all_pandora = []
    model.eval()
    with torch.no_grad():
        for x_2d, x_3d, scd, true_vcd, ux_3d, tempo_no2, _ in test_loader:
            x_2d = x_2d.to(device)
            x_3d = x_3d.to(device)
            scd = scd.to(device).squeeze(-1)
            true_vcd = true_vcd.to(device).squeeze(-1)
            ux_3d = ux_3d.to(device)

            tempo_no2 = tempo_no2.squeeze(-1)

            model_amf = model(x_2d, x_3d).squeeze(-1)

            model_amf = torch.nan_to_num(model_amf, nan=0.1, posinf=1e5, neginf=0.1)
            model_amf = torch.clamp(model_amf, min=0.01, max=5)
            model_vcd = scd / model_amf

            all_preds.append(model_vcd.detach().cpu().numpy())
            all_pandora.append(true_vcd.detach().cpu().numpy())
            all_tempo.append(tempo_no2)

    values = {
        "preds": np.concatenate(all_preds),
        "pandora": np.concatenate(all_pandora),
        "tempo": np.concatenate(all_tempo),
    }

    # Unnormalize
    mmin = norm_stats["pandora_no2_smooth"]["min"]
    mmax = norm_stats["pandora_no2_smooth"]["max"]

    preds = values["preds"] * (mmax - mmin) + mmin
    pandora = values["pandora"]

    tempo = values["tempo"]

    r2_model = pearsonr(preds, pandora)[0] ** 2
    r2_tempo = pearsonr(tempo, pandora)[0] ** 2

    # Mean Absolute Bias (MAB)
    mab_model = np.mean(np.abs(preds - pandora))
    mab_tempo = np.mean(np.abs(tempo - pandora))

    # Root Mean Squared Error (RMSE)
    rmse_model = np.sqrt(mean_squared_error(preds, pandora))
    rmse_tempo = np.sqrt(mean_squared_error(tempo, pandora))

    # Normalized MAB and RMSE (using Mean Normalization)
    target_mean = np.mean(pandora)
    nmab_model = mab_model / target_mean if target_mean != 0 else np.nan
    nmab_tempo = mab_tempo / target_mean if target_mean != 0 else np.nan
    nrmse_model = rmse_model / target_mean if target_mean != 0 else np.nan
    nrmse_tempo = rmse_tempo / target_mean if target_mean != 0 else np.nan

    ioa_model = IOA_Loss_np(pandora, preds)
    ioa_tempo = IOA_Loss_np(pandora, tempo)

    slope_model, intercept, r_value, p_value, std_err = linregress(pandora, preds)
    slope_tempo, intercept, r_value, p_value, std_err = linregress(pandora, tempo)
    metrics = {
        "n": len(pandora),
        "r2_model": r2_model,
        "r2_tempo": r2_tempo,
        "slope_model": slope_model,
        "slope_tempo": slope_tempo,
        "ioa_model": ioa_model,
        "ioa_tempo": ioa_tempo,
        "mab_model": mab_model / (1e15),
        "mab_tempo": mab_tempo / (1e15),
        "rmse_model": rmse_model / (1e15),
        "rmse_tempo": rmse_tempo / (1e15),
        "nmab_model": nmab_model,
        "nmab_tempo": nmab_tempo,
        "nrmse_model": nrmse_model,
        "nrmse_tempo": nrmse_tempo,
    }

    fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
    plot_scatter_with_fit(
        ax[0],
        pandora,
        preds,
        r"Pandora NO$_2$ column molecules/cm$^2$",
        r"Model NO$_2$ column molecules/cm$^2$",
        "Pandora vs Model",
        fig=fig,
        metrics=metrics,
        var="model",
    )
    plot_scatter_with_fit(
        ax[1],
        pandora,
        tempo,
        r"Pandora NO$_2$ column molecules/cm$^2$",
        r"Model NO$_2$ column molecules/cm$^2$",
        "Pandora vs Tempo",
        fig=fig,
        metrics=metrics,
        var="tempo",
    )

    return metrics, fig


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = Args()

    model = AMFPredictor_FNO(
        input_dim_2d=len(args.vars_2d),
        input_dim_3d=len(args.vars_3d),
        hidden_dim=128,
        n_modes=12,
        fno_hidden=16,
        dropout_prob=0.3,
    ).to(device)

    test_ds = xr.open_dataset("data/eval_sample.nc")
    norm_path = f"data/norm.pth"
    model_path = f"data/trained_models/K_fold_005.pth"
    metrics, fig = eval_model(test_ds, model_path, norm_path)
    fig.savefig("sample_eval.png", bbox_inches="tight", dpi=300)
    print(metrics)
