import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import linregress


def IOA_Loss_np(o, p):
    ioa = 1 - (np.sum((o - p) ** 2)) / (
        np.sum((np.abs(p - np.mean(o)) + np.abs(o - np.mean(o))) ** 2)
    )
    return ioa


def remove_outlier(targets, preds, tempo):
    """Removes only the single most extreme value in tempo and aligns all arrays."""

    # Find index of the maximum value in tempo
    max_index = np.argmax(tempo)

    # Remove this index from all arrays
    targets = np.delete(targets, max_index)
    preds = np.delete(preds, max_index)
    tempo = np.delete(tempo, max_index)

    return targets, preds, tempo


def calculate_metrics(values, norm_stats, plot=False, fold=None, station=None):
    mmin = norm_stats["pandora_no2_smooth"]["min"]
    mmax = norm_stats["pandora_no2_smooth"]["max"]

    preds = values["preds"] * (mmax - mmin) + mmin
    targets = values["targets"] * (mmax - mmin) + mmin
    tempo = values["tempo"]

    # preds, targets, tempo = remove_outlier(targets, preds, tempo)

    # scaler_tempo = norm['tempo']

    # tempo = scaler_tempo.inverse_transform(tempo.reshape(-1, 1)).flatten()

    # R-squared
    r2_model = pearsonr(preds, targets)[0] ** 2
    r2_tempo = pearsonr(tempo, targets)[0] ** 2

    # Mean Absolute Bias (MAB)
    mab_model = np.mean(np.abs(preds - targets))  # Calculate MAB for model
    mab_tempo = np.mean(np.abs(tempo - targets))  # Calculate MAB for tempo

    # Root Mean Squared Error (RMSE)
    rmse_model = np.sqrt(mean_squared_error(preds, targets))
    rmse_tempo = np.sqrt(mean_squared_error(tempo, targets))

    # Normalized MAB and RMSE (using Mean Normalization)
    target_mean = np.mean(targets)
    nmab_model = mab_model / target_mean if target_mean != 0 else np.nan
    nmab_tempo = mab_tempo / target_mean if target_mean != 0 else np.nan
    nrmse_model = rmse_model / target_mean if target_mean != 0 else np.nan
    nrmse_tempo = rmse_tempo / target_mean if target_mean != 0 else np.nan

    ioa_model = IOA_Loss_np(targets, preds)
    ioa_tempo = IOA_Loss_np(targets, tempo)

    metrics = {
        "Station": station,
        "n": len(targets),
        "fold": fold,
        "r2_model": r2_model,
        "r2_tempo": r2_tempo,
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

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
        plot_scatter_with_fit(
            ax[0],
            targets,
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
            targets,
            tempo,
            r"Pandora NO$_2$ column molecules/cm$^2$",
            r"Model NO$_2$ column molecules/cm$^2$",
            "Pandora vs Tempo",
            fig=fig,
            metrics=metrics,
            var="tempo",
        )
        if station:
            fig.suptitle(f"Station: {station}")
        else:
            fig.suptitle(f"Fold: {fold}")
        plt.subplots_adjust(top=0.95)

        return metrics, fig
    else:
        return metrics


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
