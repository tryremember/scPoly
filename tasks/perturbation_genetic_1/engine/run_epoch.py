import os
import time

import torch


def train_epoch(model, dataloader, device, optimizer, moe, epoch, log_interval, save_dir):
    model.train()
    start_time = time.time()
    n_batch = len(dataloader)

    total_loss = 0.0
    total_aux_loss = 0.0
    total_mse_all = 0.0
    total_mse_de = 0.0
    total_pearson_all = 0.0
    total_pearson_de = 0.0
    total_delta_pearson_de = 0.0

    for batch, batch_data in enumerate(dataloader):
        batch_data = {k: v.to(model.device) for k, v in batch_data.items()}
        output = model(**batch_data)

        loss = output["total_loss"]
        aux_loss = output["l_aux"]
        mse_all = output["mse_loss_all"]
        mse_de = output["mse_loss_de"]
        pearson_all = output["pearson_all"]
        pearson_de = output["pearson_de"]
        delta_pearson_de = output["delta_pearson_de"]

        total_aux_loss += aux_loss
        total_mse_all += mse_all
        total_loss += loss.item()
        total_mse_de += mse_de
        total_pearson_all += pearson_all
        total_pearson_de += pearson_de
        total_delta_pearson_de += delta_pearson_de

        if moe:
            model.backward(loss)
            model.step()
        else:
            model.zero_grad()
            loss.backward()
            optimizer.step()

        if batch % log_interval == 0 and model.local_rank == 0:
            avg_loss = total_loss / (batch + 1)
            avg_aux_loss = total_aux_loss / (batch + 1)
            avg_mse_all = total_mse_all / (batch + 1)
            avg_mse_de = total_mse_de / (batch + 1)
            avg_pearson_all = total_pearson_all / (batch + 1)
            avg_pearson_de = total_pearson_de / (batch + 1)
            avg_delta_pearson_de = total_delta_pearson_de / (batch + 1)

            print(
                f"Epoch {epoch} | Batch {batch}/{n_batch} | "
                f"Avg total Loss: {avg_loss:.4f} | "
                f"Avg aux Loss: {avg_aux_loss:.4f} | "
                f"Avg mse all: {avg_mse_all:.4f} | "
                f"Avg mse de: {avg_mse_de:.4f} | "
                f"Avg pearson all: {avg_pearson_all:.4f} | "
                f"Avg pearson de: {avg_pearson_de:.4f} | "
                f"Avg delta pearson de: {avg_delta_pearson_de:.4f} | "
            )

    avg_loss = total_loss / n_batch
    avg_mse_all = total_mse_all / n_batch
    avg_aux_loss = total_aux_loss / n_batch
    avg_mse_de = total_mse_de / n_batch
    avg_pearson_all = total_pearson_all / n_batch
    avg_pearson_de = total_pearson_de / n_batch
    avg_delta_pearson_de = total_delta_pearson_de / n_batch

    epoch_time = time.time() - start_time
    if model.local_rank == 0:
        print(
            f"Epoch {epoch} completed in {epoch_time:.2f} seconds | "
            f"Avg total Loss: {avg_loss:.4f} | "
            f"Avg aux Loss: {avg_aux_loss:.4f} | "
            f"Avg mse all: {avg_mse_all:.4f} | "
            f"Avg mse de: {avg_mse_de:.4f} | "
            f"Avg pearson all: {avg_pearson_all:.4f} | "
            f"Avg pearson de: {avg_pearson_de:.4f} | "
            f"Avg delta pearson de: {avg_delta_pearson_de:.4f} | "
        )


def valid_epoch(model, dataloader, device, epoch, log_interval, save_dir):
    model.eval()
    start_time = time.time()
    n_batch = len(dataloader)

    total_loss = 0.0
    total_aux_loss = 0.0
    total_mse_all = 0.0
    total_mse_de = 0.0
    total_pearson_all = 0.0
    total_pearson_de = 0.0
    total_delta_pearson_de = 0.0

    with torch.no_grad():
        for batch, batch_data in enumerate(dataloader):
            batch_data = {k: v.to(model.device) for k, v in batch_data.items()}
            output = model(inf=False, **batch_data)

            loss = output["total_loss"]
            aux_loss = output["l_aux"]
            mse_all = output["mse_loss_all"]
            mse_de = output["mse_loss_de"]
            pearson_all = output["pearson_all"]
            pearson_de = output["pearson_de"]
            delta_pearson_de = output["delta_pearson_de"]

            total_aux_loss += aux_loss
            total_mse_all += mse_all
            total_loss += loss.item()
            total_mse_de += mse_de
            total_pearson_all += pearson_all
            total_pearson_de += pearson_de
            total_delta_pearson_de += delta_pearson_de

            if batch % log_interval == 0 and model.local_rank == 0:
                avg_loss = total_loss / (batch + 1)
                avg_aux_loss = total_aux_loss / (batch + 1)
                avg_mse_all = total_mse_all / (batch + 1)
                avg_mse_de = total_mse_de / (batch + 1)
                avg_pearson_all = total_pearson_all / (batch + 1)
                avg_pearson_de = total_pearson_de / (batch + 1)
                avg_delta_pearson_de = total_delta_pearson_de / (batch + 1)

                print(
                    f"Epoch {epoch} | Batch {batch}/{n_batch} | "
                    f"Avg total Loss: {avg_loss:.4f} | "
                    f"Avg aux Loss: {avg_aux_loss:.4f} | "
                    f"Avg mse all: {avg_mse_all:.4f} | "
                    f"Avg mse de: {avg_mse_de:.4f} | "
                    f"Avg pearson all: {avg_pearson_all:.4f} | "
                    f"Avg pearson de: {avg_pearson_de:.4f} | "
                    f"Avg delta pearson de: {avg_delta_pearson_de:.4f} | "
                )

    avg_loss = total_loss / n_batch
    avg_mse_all = total_mse_all / n_batch
    avg_aux_loss = total_aux_loss / n_batch
    avg_mse_de = total_mse_de / n_batch
    avg_pearson_all = total_pearson_all / n_batch
    avg_pearson_de = total_pearson_de / n_batch
    avg_delta_pearson_de = total_delta_pearson_de / n_batch

    epoch_time = time.time() - start_time
    if model.local_rank == 0:
        print(
            f"Epoch {epoch} completed in {epoch_time:.2f} seconds | "
            f"Avg total Loss: {avg_loss:.4f} | "
            f"Avg aux Loss: {avg_aux_loss:.4f} | "
            f"Avg mse all: {avg_mse_all:.4f} | "
            f"Avg mse de: {avg_mse_de:.4f} | "
            f"Avg pearson all: {avg_pearson_all:.4f} | "
            f"Avg pearson de: {avg_pearson_de:.4f} | "
            f"Avg delta pearson de: {avg_delta_pearson_de:.4f} | "
        )

    return avg_mse_all


def inf_epoch(model, dataloader, device, save_dir):
    model.eval()
    start_time = time.time()

    ctrl_exp_de_list = []
    target_exp_de_list = []
    pred_exp_de_list = []

    with torch.no_grad():
        for batch, batch_data in enumerate(dataloader):
            batch_data = {k: v.to(model.device) for k, v in batch_data.items()}
            output = model(inf=True, **batch_data)

            ctrl_exp_de_list.append(output["ctrl_exp_de"])
            target_exp_de_list.append(output["target_exp_de"])
            pred_exp_de_list.append(output["pred_exp_de"])

    ctrl_exp_de_all = torch.cat(ctrl_exp_de_list, dim=0)
    target_exp_de_all = torch.cat(target_exp_de_list, dim=0)
    pred_exp_de_all = torch.cat(pred_exp_de_list, dim=0)

    os.makedirs(save_dir, exist_ok=True)
    save_path_ctrl_exp_de = os.path.join(save_dir, "ctrl_exp_de.pt")
    save_path_target_exp_de = os.path.join(save_dir, "target_exp_de.pt")
    save_path_pred_exp_de = os.path.join(save_dir, "pred_exp_de.pt")

    torch.save(ctrl_exp_de_all, save_path_ctrl_exp_de)
    torch.save(target_exp_de_all, save_path_target_exp_de)
    torch.save(pred_exp_de_all, save_path_pred_exp_de)

    epoch_time = time.time() - start_time
    if model.local_rank == 0:
        print(f"Inference done in {epoch_time} seconds.")
        print(f"pred_exp_value_de saved to {save_path_pred_exp_de}")

    return ctrl_exp_de_all, target_exp_de_all, pred_exp_de_all
