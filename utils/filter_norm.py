import pandas as pd
import xarray as xr
import numpy as np
import pickle
# import matplotlib.pyplot as plt


def find_outliers(data_array, method="percentile", threshold=2.5, percentile=1):
    if method == "iqr":
        # IQR method
        q1 = np.percentile(data_array.values, 25)
        q3 = np.percentile(data_array.values, 75)
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

    elif method == "std":
        # Standard deviation method
        mean = np.mean(data_array.values)
        std = np.std(data_array.values)

        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std

    elif method == "percentile":
        # Percentile method
        lower_bound = np.percentile(data_array.values, percentile)
        upper_bound = np.percentile(data_array.values, 100 - percentile)

    # Create outlier mask
    is_outlier = (data_array < lower_bound) | (data_array > upper_bound)

    return is_outlier


def normalize_train(ds, vars_norm):
    coords = {coord: ds.coords[coord] for coord in ds.coords}
    ds_norm = xr.Dataset(coords=coords)

    normalization_stats = {}
    for var in vars_norm["3d"]:
        ds_norm[f"{var}_orig"] = ds[f"{var}"].copy(deep=True)

    # Normalize 2D variables
    for var in vars_norm["2d"]:
        mean_layer = ds[var].mean(dim="time")
        std_layer = ds[var].std(dim="time")
        ds_norm[f"{var}_norm"] = (ds[var] - mean_layer) / std_layer
        normalization_stats[var] = {"mean": mean_layer, "std": std_layer}

    # Normalize 3D profile variables separately for each layer
    for var in vars_norm["3d"]:
        mean_layer = ds[var].mean(dim="time")
        std_layer = ds[var].std(dim="time")
        ds_norm[f"{var}_norm"] = (ds[var] - mean_layer) / std_layer
        normalization_stats[var] = {"mean": mean_layer, "std": std_layer}

    # Normalize angle variables using sine-cosine transformation
    for angle in vars_norm["angles"]:
        angle_rad = np.deg2rad(ds[angle])
        ds_norm[f"{angle}_sin"] = np.sin(angle_rad)
        ds_norm[f"{angle}_cos"] = np.cos(angle_rad)

    # Normalize time variables using sine-cosine transformation for cyclical encoding
    ds["time"] = pd.to_datetime(ds["time"].values)
    ds_norm["hour_sin"] = np.sin(2 * np.pi * ds["time"].dt.hour / 24)
    ds_norm["hour_cos"] = np.cos(2 * np.pi * ds["time"].dt.hour / 24)
    ds_norm["day_of_year_sin"] = np.sin(2 * np.pi * ds["time"].dt.dayofyear / 365)
    ds_norm["day_of_year_cos"] = np.cos(2 * np.pi * ds["time"].dt.dayofyear / 365)

    for var in vars_norm["no2"]:
        min_val = ds[var].min().values
        max_val = ds[var].max().values
        normalization_stats[var] = {"min": min_val, "max": max_val}
        ds_norm[f"{var}_minmax_scaled"] = (ds[var] - min_val) / (max_val - min_val)

    # Normalize latitude and longitude with Min-Max scaling

    lat_min, lat_max = 10, 60  # Latitude: 10°N to 60°N
    lon_min, lon_max = -140, -60

    ds_norm["lat_norm"] = (ds["latitude"] - lat_min) / (lat_max - lat_min)
    ds_norm["lon_norm"] = (ds["longitude"] - lon_min) / (lon_max - lon_min)

    normalization_stats["lat"] = {"min": lat_min, "max": lat_max}
    normalization_stats["lon"] = {"min": lon_min, "max": lon_max}

    ds_norm["Tempo_NO2_total"] = ds["Tempo_NO2_total"].copy()
    ds_norm["pandora_no2_smooth"] = ds["pandora_no2_smooth"].copy()

    return ds_norm, normalization_stats


def normalize_test(
    ds_new, vars_norm, norm_stats=None, stats_file="normalization_stats.pkl"
):
    # Load normalization statistics
    if norm_stats is None:
        with open(stats_file, "rb") as f:
            normalization_stats = pickle.load(f)
    else:
        normalization_stats = norm_stats

    coords = {coord: ds_new.coords[coord] for coord in ds_new.coords}
    ds_new_norm = xr.Dataset(coords=coords)

    for var in vars_norm["3d"]:
        ds_new_norm[f"{var}_orig"] = ds_new[f"{var}"].copy(deep=True)

    # Normalize 2D variables
    for var in vars_norm["2d"]:
        mean_layer = normalization_stats[var]["mean"]
        std_layer = normalization_stats[var]["std"]
        ds_new_norm[f"{var}_norm"] = (ds_new[var] - mean_layer) / std_layer

    # Normalize 3D profile variables separately for each layer
    for var in vars_norm["3d"]:
        mean_layer = normalization_stats[var]["mean"]
        std_layer = normalization_stats[var]["std"]
        ds_new_norm[f"{var}_norm"] = (ds_new[var] - mean_layer) / std_layer

    # Normalize angle variables using sine-cosine transformation
    for angle in vars_norm["angles"]:
        angle_rad = np.deg2rad(ds_new[angle])
        ds_new_norm[f"{angle}_sin"] = np.sin(angle_rad)
        ds_new_norm[f"{angle}_cos"] = np.cos(angle_rad)

    # Normalize time variables using sine-cosine transformation for cyclical encoding
    ds_new["time"] = pd.to_datetime(ds_new["time"].values)
    ds_new_norm["hour_sin"] = np.sin(2 * np.pi * ds_new["time"].dt.hour / 24)
    ds_new_norm["hour_cos"] = np.cos(2 * np.pi * ds_new["time"].dt.hour / 24)
    ds_new_norm["day_of_year_sin"] = np.sin(
        2 * np.pi * ds_new["time"].dt.dayofyear / 365
    )
    ds_new_norm["day_of_year_cos"] = np.cos(
        2 * np.pi * ds_new["time"].dt.dayofyear / 365
    )

    for var in vars_norm["no2"]:
        min_val = normalization_stats[var]["min"]
        max_val = normalization_stats[var]["max"]
        ds_new_norm[f"{var}_minmax_scaled"] = (ds_new[var] - min_val) / (
            max_val - min_val
        )

    # Normalize latitude and longitude with saved Min-Max scaling
    lat_min, lat_max = (
        normalization_stats["lat"]["min"],
        normalization_stats["lat"]["max"],
    )
    lon_min, lon_max = (
        normalization_stats["lon"]["min"],
        normalization_stats["lon"]["max"],
    )

    ds_new_norm["lat_norm"] = (ds_new["latitude"] - lat_min) / (lat_max - lat_min)
    ds_new_norm["lon_norm"] = (ds_new["longitude"] - lon_min) / (lon_max - lon_min)
    ds_new_norm["Tempo_NO2_total"] = ds_new["Tempo_NO2_total"].copy()
    ds_new_norm["pandora_no2_smooth"] = ds_new["pandora_no2_smooth"].copy()

    return ds_new_norm


def unnormalize_no2_vcd(preds, norm_stats=None, stats_file="normalization_stats.pkl"):
    """Unnormalize predicted NO2 VCD using saved statistics from Pandora NO2"""

    # Load normalization statistics
    if norm_stats is None:
        with open(stats_file, "rb") as f:
            normalization_stats = pickle.load(f)
    else:
        normalization_stats = norm_stats

    # Get mean and std from Pandora NO2 statistics
    mean_log = normalization_stats["pandora_no2_smooth"]["mean"]
    std_log = normalization_stats["pandora_no2_smooth"]["std"]

    # Reverse normalization (standardization) and log transformation
    unnormalized_no2 = np.expm1((preds * std_log) + mean_log)

    return unnormalized_no2


def prep_data(ds, test=False, norm_stats=None, args=None):
    vars_angle = [
        "solar_zenith_angle",
        "solar_azimuth_angle",
        "viewing_zenith_angle",
        "viewing_azimuth_angle",
        "relative_azimuth_angle",
    ]

    vars_3d = {
        "scattering_weights": ("time", "swt_level"),
        "gas_profile": ("time", "swt_level"),
        "temperature_profile": ("time", "swt_level"),
    }

    vars_2d = [
        "vertical_column_troposphere",
        "vertical_column_stratosphere",
        "vertical_column_total",
        "snow_ice_fraction",
        "terrain_height",
        "surface_pressure",
        "albedo",
        "eff_cloud_fraction",
    ]

    vars_no2 = [
        "fitted_slant_column",
        "pandora_no2_smooth",
    ]
    vars_norm = {"angles": vars_angle, "3d": vars_3d, "2d": vars_2d, "no2": vars_no2}

    combined_ds = ds.copy()

    if test:
        # ds_filtered = combined_ds.copy()

        ds_norm = normalize_test(combined_ds, vars_norm, norm_stats)
        for var in combined_ds.data_vars:
            if var not in vars_norm:
                ds_norm[var] = combined_ds[var]

    else:
        outliers_slant = find_outliers(
            combined_ds.fitted_slant_column, method="percentile", percentile=1
        )
        outliers_pandora = find_outliers(
            combined_ds.pandora_no2_smooth, method="percentile", percentile=1
        )

        # Combine masks - a time point is considered an outlier if either variable has an outlier
        combined_outlier_mask = outliers_slant | outliers_pandora
        is_not_outlier = ~combined_outlier_mask

        # Create filtered dataset
        ds_filtered = combined_ds.isel(time=is_not_outlier)
        # ds_filtered =  combined_ds.copy()
        ds_norm, norm_stats = normalize_train(ds_filtered, vars_norm)

        for var in ds_filtered.data_vars:
            if var not in vars_norm:
                ds_norm[var] = ds_filtered[var]

    return ds_norm, norm_stats
