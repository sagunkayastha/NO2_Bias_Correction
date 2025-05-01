import numpy as np
import torch
from tqdm import tqdm
from loss import amf_loss


def epoch_train(model, train_loader, optimizer, criterion, device, lambda_physics=0.1):
    """
    Trains the model for one epoch with a tqdm progress bar.

    Args:
        model: The model to train.
        train_loader: DataLoader for the training data.
        optimizer: Optimizer for training.
        criterion: Loss function.
        device: Device to use (CPU or GPU).
        lambda_physics: Weight for the physics-based loss.

    Returns:
        model: The trained model.
        Average training loss for the epoch.
        Average physics loss for the epoch.
        Average VCD loss for the epoch.
    """
    model.train()  # Set the model to training mode
    total_loss = 0.0
    physics_loss_epoch = 0.0
    vcd_loss_epoch = 0.0
    amf_ll_epoch = 0.0
    num_batches = 0
    preds, targets, tempo, all_amf = [], [], [], []
    tempo_amfs, model_amfs = [], []
    # Wrap the train_loader with tqdm

    with tqdm(train_loader, unit="batch") as tepoch:
        for i, (x_2d, x_3d, scd, true_vcd, ux_3d, tempo_no2, tempo_amf, _) in enumerate(
            tepoch
        ):
            x_2d = x_2d.to(device)
            x_3d = x_3d.to(device)
            scd = scd.to(device).squeeze(-1)
            true_vcd = true_vcd.to(device).squeeze(-1)
            ux_3d = ux_3d.to(device).squeeze(-1)
            orig_amf = tempo_amf.to(device).squeeze(-1)

            optimizer.zero_grad()

            model_amf = model(x_2d, x_3d).squeeze(-1)

            model_amf = torch.nan_to_num(model_amf, nan=0.1, posinf=1e5, neginf=0.1)
            model_amf = torch.clamp(model_amf, min=0.01)

            amf_ll = criterion(model_amf, orig_amf)

            model_vcd = scd.squeeze(-1) / model_amf
            vcd_loss = criterion(model_vcd, true_vcd)

            physics_loss = amf_loss(model_amf, ux_3d, criterion)
            # physics_loss = amf_ll
            combined_loss = vcd_loss + lambda_physics * physics_loss

            combined_loss.backward()
            # amf_ll.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            optimizer.step()

            total_loss += combined_loss.item()
            physics_loss_epoch += physics_loss.item()
            vcd_loss_epoch += vcd_loss.item()
            amf_ll_epoch += amf_ll.item()
            num_batches += 1

            preds.append(model_vcd.detach().cpu().numpy())
            targets.append(true_vcd.detach().cpu().numpy())
            tempo.append(tempo_no2)
            model_amfs.append(model_amf.detach().cpu().numpy())
            tempo_amfs.append(tempo_amf)

            # Update tqdm progress bar

            if (i + 10) % 10 == 0:
                tepoch.set_postfix(
                    total_loss=combined_loss.item(),
                    vcd_loss=vcd_loss.item(),
                    physics_loss=physics_loss.item(),
                )
    values = {
        "preds": np.concatenate(preds),
        "targets": np.concatenate(targets),
        "tempo": np.concatenate(tempo),
    }

    return (
        model,
        total_loss / num_batches,
        physics_loss_epoch / num_batches,
        vcd_loss_epoch / num_batches,
        values,
        [tempo_amfs, model_amfs],
    )


def epoch_eval(model, test_loader, criterion, device, lambda_physics=0.1):
    """
    Evaluates the model on the test loader with a tqdm progress bar.

    Args:
        model: The model to evaluate.
        test_loader: DataLoader for the test data.
        criterion: Loss function.
        device: Device to use (CPU or GPU).
        lambda_physics: Weight for the physics-based loss.

    Returns:
        Average total loss for the test set.
        Average physics loss for the test set.
        Average VCD loss for the test set.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    physics_loss_epoch = 0.0
    vcd_loss_epoch = 0.0
    amf_ll_epoch = 0.0
    all_scd = []
    all_dates = []
    num_batches = 0
    preds, targets, tempo = [], [], []
    tempo_amfs, model_amfs = [], []
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            for i, (
                x_2d,
                x_3d,
                scd,
                true_vcd,
                ux_3d,
                tempo_no2,
                tempo_amf,
                dates,
            ) in enumerate(tepoch):
                x_2d = x_2d.to(device)
                x_3d = x_3d.to(device)
                scd = scd.to(device).squeeze(-1)
                true_vcd = true_vcd.to(device).squeeze(-1)
                ux_3d = ux_3d.to(device)
                orig_amf = tempo_amf.to(device).squeeze(-1)

                tempo_no2 = tempo_no2.squeeze(-1)

                model_amf = model(x_2d, x_3d).squeeze(-1)

                model_amf = torch.nan_to_num(model_amf, nan=0.1, posinf=1e5, neginf=0.1)
                model_amf = torch.clamp(model_amf, min=0.01, max=10)
                model_vcd = scd / model_amf

                vcd_loss = criterion(model_vcd, true_vcd)

                physics_loss = amf_loss(model_amf, ux_3d, criterion)
                # amf_ll = criterion(model_amf, orig_amf)
                # physics_loss = criterion(model_amf, orig_amf)
                combined_loss = vcd_loss + lambda_physics * physics_loss

                total_loss += combined_loss.item()
                physics_loss_epoch += physics_loss.item()
                vcd_loss_epoch += vcd_loss.item()
                num_batches += 1

                # if (i + 10) % 10 == 0:
                #     tepoch.set_postfix(
                #         total_loss=combined_loss.item(),
                #         vcd_loss=vcd_loss.item(),
                #         physics_loss=physics_loss.item(),
                #     )

                preds.append(model_vcd.detach().cpu().numpy())
                targets.append(true_vcd.detach().cpu().numpy())
                tempo.append(tempo_no2)
                model_amfs.append(model_amf.detach().cpu().numpy())
                tempo_amfs.append(tempo_amf)
                all_scd.append(scd.detach().cpu().numpy())
                all_dates.append(dates.numpy())
    dates = np.concatenate(all_dates)
    values = {
        "preds": np.concatenate(preds),
        "targets": np.concatenate(targets),
        "tempo": np.concatenate(tempo),
    }
    return (
        total_loss / num_batches,
        physics_loss_epoch / num_batches,
        vcd_loss_epoch / num_batches,
        values,
        [tempo_amfs, model_amfs, all_scd],
        all_dates,
    )
