# Replace argparse with a manual setup
import os


class Args:
    def __init__(self):
        self.run_name = "k_fold"
        self.run_folder = os.getcwd()
        self.collocated_data_path = "data/combined_ds.nc"
        self.details_file = "data/final_details.csv"
        self.num_epochs = 40
        self.lr = 1e-4
        self.lambda_physics = 0.005
        self.batch_size = 128
        self.vars_2d = [
            "fitted_slant_column_minmax_scaled",
            "snow_ice_fraction_norm",
            "terrain_height_norm",
            "surface_pressure_norm",
            "albedo_norm",
            "eff_cloud_fraction_norm",
            "solar_zenith_angle_sin",
            "solar_zenith_angle_cos",
            "solar_azimuth_angle_sin",
            "solar_azimuth_angle_cos",
            "viewing_zenith_angle_sin",
            "viewing_zenith_angle_cos",
            "viewing_azimuth_angle_sin",
            "viewing_azimuth_angle_cos",
            "relative_azimuth_angle_sin",
            "relative_azimuth_angle_cos",
            "hour_sin",
            "hour_cos",
            "day_of_year_sin",
            "day_of_year_cos",
            "lat_norm",
            "lon_norm",
        ]

        self.vars_3d = [
            "scattering_weights_norm",
            "gas_profile_norm",
            "temperature_profile_norm",
        ]

        self.scd_var = "fitted_slant_column_minmax_scaled"  # NO2 Slant Column
        self.vcd_target = "pandora_no2_smooth_minmax_scaled"  # NO2 Pandora VCD
