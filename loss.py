import torch
import numpy as np
import torch.nn.functional as F


def IOA_Loss(o, p):
    ioa = 1 - (torch.sum((o - p) ** 2)) / (
        torch.sum((torch.abs(p - torch.mean(o)) + torch.abs(o - torch.mean(o))) ** 2)
    )
    return -ioa


def pearson_corrcoef(x, y, eps=1e-6):
    """
    x: First set of values (predicted)
    y: Second set of values (ground truth)
    eps: Small constant to prevent division by zero
    """
    x_mean = x.mean()
    y_mean = y.mean()

    cov = ((x - x_mean) * (y - y_mean)).mean()
    x_std = x.std() + eps
    y_std = y.std() + eps

    return cov / (x_std * y_std)


def Huber_loss(predicted_vcd, target_vcd, delta=1.0):
    """Huber loss for VCD.

    Args:
        predicted_vcd (torch.Tensor): The predicted VCD values.
        target_vcd (torch.Tensor): The target VCD values.
        delta (float, optional):  Parameter for Huber loss.
            Defaults to 1.0.
    Returns:
        torch.Tensor: The Huber loss.
    """
    loss = F.smooth_l1_loss(predicted_vcd, target_vcd, beta=delta)
    return loss


def compute_loss(
    model_amf,
    scd,
    true_vcd,
    scattering_weights,
    gas_profile,
    temperature_profile,
    norm_factors,
    criterion,
):
    """
    Computes the total loss:
    - NO₂ VCD loss (MSE)
    - AMF physics loss (ensuring model respects the physics equation)

    Parameters:
    - model_amf: Model-predicted AMF
    - scd: Fitted slant column density (from dataset)
    - true_vcd: True NO₂ column (pandora_no2_smooth)
    - scattering_weights: Model input
    - gas_profile: Model input
    - temperature_profile: Model input
    - norm_factors: Normalization factors for inverse transforming inputs
    - lambda_physics: Weight for physics loss

    Returns:
    - total_loss: Weighted sum of NO₂ column loss and AMF physics loss
    """

    # Inverse normalize inputs
    sw_min, sw_max = (
        norm_factors["scattering_weights"]["min"],
        norm_factors["scattering_weights"]["max"],
    )
    gp_min, gp_max = (
        norm_factors["gas_profile"]["min"],
        norm_factors["gas_profile"]["max"],
    )
    tp_min, tp_max = (
        norm_factors["temperature_profile"]["min"],
        norm_factors["temperature_profile"]["max"],
    )
    # print(sw_max, sw_min, gp_max, gp_min, tp_max, tp_min)
    scattering_weights = scattering_weights * (sw_max - sw_min) + sw_min
    gas_profile = gas_profile * (gp_max - gp_min) + gp_min
    temperature_profile = temperature_profile * (tp_max - tp_min) + tp_min

    # Compute shape factor S(z)
    shape_factor = gas_profile / gas_profile.sum(dim=1, keepdim=True)

    # Compute temperature correction c(z)
    a, b, T_sigma = 0.00316, 3.39e-6, 220  # Constants from TEMPO documentation
    temp_correction = (
        1
        - a * (temperature_profile - T_sigma)
        + b * (temperature_profile - T_sigma) ** 2
    )

    # Compute AMF using physics equation
    physics_amf = (scattering_weights * shape_factor * temp_correction).sum(dim=1)

    # Compute NO₂ column loss (MSE)
    model_vcd = scd / model_amf

    vcd_loss = criterion(model_vcd, true_vcd)

    # Compute AMF physics loss
    physics_loss = criterion(model_amf, physics_amf)

    # vcd_loss = vcd_loss / vcd_loss.detach().mean()
    # physics_loss = physics_loss / physics_loss.detach().mean()

    return vcd_loss, physics_loss


def amf_loss(model_amf, ux_3d, criterion):
    scattering_weights, gas_profile, temperature_profile = (
        ux_3d[:, 0, :],
        ux_3d[:, 1, :],
        ux_3d[:, 2, :],
    )
    shape_factor = gas_profile / gas_profile.sum(dim=1, keepdim=True)

    # Compute temperature correction c(z)
    a, b, T_sigma = 0.00316, 3.39e-6, 220
    temp_correction = (
        1
        - a * (temperature_profile - T_sigma)
        + b * (temperature_profile - T_sigma) ** 2
    )

    # Compute AMF using physics equation
    physics_amf = (scattering_weights * shape_factor * temp_correction).sum(dim=1)
    physics_loss = criterion(model_amf, physics_amf)

    return physics_loss


def asymmetric_loss(predictions, targets, underestimation_penalty=2.0):
    """
    Penalizes underestimations more than overestimations.
    """
    diff = predictions - targets
    loss = torch.where(
        diff < 0, underestimation_penalty * torch.abs(diff), torch.abs(diff)
    )
    return torch.mean(loss)
