import os
import time

import torch


def train_epoch(model, dataloader, device, optimizer, moe, epoch, log_interval, save_dir):
    del device, save_dir
    model.train()
    start_time = time.time()
    n_batch = len(dataloader)

    total_loss = 0.0
    total_aux_loss = 0.0
    total_mse = 0.0
    total_l2_loss = 0.0
    total_pearson = 0.0
    total_delta_pearson = 0.0
    total_spearman = 0.0
    total_delta_spearman = 0.0
    for batch, batch_data in enumerate(dataloader):
        batch_data = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in batch_data.items()}
        output = model(**batch_data)

        loss = output["total_loss"]
        aux_loss = output["l_aux"]
        l2_loss = output["l2_loss"]
        mse = output["mse"]
        pearson = output["pearson"]
        delta_pearson = output["delta_pearson"]
        spearman = output["spearman"]
        delta_spearman = output["delta_spearman"]
        total_aux_loss += aux_loss
        total_mse += mse
        total_loss += loss.item()
        total_l2_loss += l2_loss
        total_pearson += pearson
        total_delta_pearson += delta_pearson
        total_spearman += spearman
        total_delta_spearman += delta_spearman

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
            avg_mse = total_mse / (batch + 1)
            avg_l2_loss = total_l2_loss / (batch + 1)
            avg_pearson = total_pearson / (batch + 1)
            avg_delta_pearson = total_delta_pearson / (batch + 1)
            avg_spearman = total_spearman / (batch + 1)
            avg_delta_spearman = total_delta_spearman / (batch + 1)
            print(
                f"Epoch {epoch} | Batch {batch}/{n_batch} | "
                f"Avg total Loss: {avg_loss:.4f} | "
                f"Avg aux Loss: {avg_aux_loss:.4f} | "
                f"Avg mse: {avg_mse:.4f} | "
                f"Avg l2 loss: {avg_l2_loss:.4f} | "
                f"Avg pearson: {avg_pearson:.4f} | "
                f"Avg delta pearson: {avg_delta_pearson:.4f} | "
                f"Avg spearman: {avg_spearman:.4f} | "
                f"Avg delta spearman: {avg_delta_spearman:.4f}"
            )

    avg_loss = total_loss / n_batch
    avg_mse = total_mse / n_batch
    avg_aux_loss = total_aux_loss / n_batch
    avg_l2_loss = total_l2_loss / n_batch
    avg_pearson = total_pearson / n_batch
    avg_delta_pearson = total_delta_pearson / n_batch
    avg_spearman = total_spearman / n_batch
    avg_delta_spearman = total_delta_spearman / n_batch
    epoch_time = time.time() - start_time
    if model.local_rank == 0:
        print(
            f"Epoch {epoch} completed in {epoch_time:.2f} seconds | "
            f"Avg total Loss: {avg_loss:.4f} | "
            f"Avg aux Loss: {avg_aux_loss:.4f} | "
            f"Avg mse: {avg_mse:.4f} | "
            f"Avg l2 loss: {avg_l2_loss:.4f} | "
            f"Avg pearson: {avg_pearson:.4f} | "
            f"Avg delta pearson: {avg_delta_pearson:.4f} | "
            f"Avg spearman: {avg_spearman:.4f} | "
            f"Avg delta spearman: {avg_delta_spearman:.4f}"
        )


def valid_epoch(model, dataloader, device, epoch, log_interval, save_dir):
    del device, save_dir
    model.eval()
    start_time = time.time()
    n_batch = len(dataloader)

    total_loss = 0.0
    total_aux_loss = 0.0
    total_mse = 0.0
    total_l2_loss = 0.0
    total_pearson = 0.0
    total_delta_pearson = 0.0
    total_spearman = 0.0
    total_delta_spearman = 0.0
    with torch.no_grad():
        for batch, batch_data in enumerate(dataloader):
            batch_data = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in batch_data.items()}
            output = model(**batch_data)

            loss = output["total_loss"]
            aux_loss = output["l_aux"]
            l2_loss = output["l2_loss"]
            mse = output["mse"]
            pearson = output["pearson"]
            delta_pearson = output["delta_pearson"]
            spearman = output["spearman"]
            delta_spearman = output["delta_spearman"]
            total_aux_loss += aux_loss
            total_mse += mse
            total_loss += loss.item()
            total_l2_loss += l2_loss
            total_pearson += pearson
            total_delta_pearson += delta_pearson
            total_spearman += spearman
            total_delta_spearman += delta_spearman

            if batch % log_interval == 0 and model.local_rank == 0:
                avg_loss = total_loss / (batch + 1)
                avg_aux_loss = total_aux_loss / (batch + 1)
                avg_mse = total_mse / (batch + 1)
                avg_l2_loss = total_l2_loss / (batch + 1)
                avg_pearson = total_pearson / (batch + 1)
                avg_delta_pearson = total_delta_pearson / (batch + 1)
                avg_spearman = total_spearman / (batch + 1)
                avg_delta_spearman = total_delta_spearman / (batch + 1)
                print(
                    f"Epoch {epoch} | Batch {batch}/{n_batch} | "
                    f"Avg total Loss: {avg_loss:.4f} | "
                    f"Avg aux Loss: {avg_aux_loss:.4f} | "
                    f"Avg mse: {avg_mse:.4f} | "
                    f"Avg l2 loss: {avg_l2_loss:.4f} | "
                    f"Avg pearson: {avg_pearson:.4f} | "
                    f"Avg delta pearson: {avg_delta_pearson:.4f} | "
                    f"Avg spearman: {avg_spearman:.4f} | "
                    f"Avg delta spearman: {avg_delta_spearman:.4f}"
                )

    avg_loss = total_loss / n_batch
    avg_mse = total_mse / n_batch
    avg_aux_loss = total_aux_loss / n_batch
    avg_l2_loss = total_l2_loss / n_batch
    avg_pearson = total_pearson / n_batch
    avg_delta_pearson = total_delta_pearson / n_batch
    avg_spearman = total_spearman / n_batch
    avg_delta_spearman = total_delta_spearman / n_batch
    epoch_time = time.time() - start_time
    if model.local_rank == 0:
        print(
            f"Epoch {epoch} completed in {epoch_time:.2f} seconds | "
            f"Avg total Loss: {avg_loss:.4f} | "
            f"Avg aux Loss: {avg_aux_loss:.4f} | "
            f"Avg mse: {avg_mse:.4f} | "
            f"Avg l2 loss: {avg_l2_loss:.4f} | "
            f"Avg pearson: {avg_pearson:.4f} | "
            f"Avg delta pearson: {avg_delta_pearson:.4f} | "
            f"Avg spearman: {avg_spearman:.4f} | "
            f"Avg delta spearman: {avg_delta_spearman:.4f}"
        )

    return avg_l2_loss


def inf_epoch(model, dataloader, device, log_interval, save_dir):
    del device
    model.eval()
    start_time = time.time()
    pred_exp_list = []
    n_batch = len(dataloader)

    total_loss = 0.0
    total_aux_loss = 0.0
    total_mse = 0.0
    total_l2_loss = 0.0
    total_pearson = 0.0
    total_delta_pearson = 0.0
    total_spearman = 0.0
    total_delta_spearman = 0.0
    with torch.no_grad():
        for batch, batch_data in enumerate(dataloader):
            batch_data = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in batch_data.items()}
            output = model(inf=True, **batch_data)

            loss = output["total_loss"]
            aux_loss = output["l_aux"]
            l2_loss = output["l2_loss"]
            mse = output["mse"]
            pearson = output["pearson"]
            delta_pearson = output["delta_pearson"]
            spearman = output["spearman"]
            delta_spearman = output["delta_spearman"]
            pred_exp_list.append(output["pred_exp"])

            total_aux_loss += aux_loss
            total_mse += mse
            total_loss += loss.item()
            total_l2_loss += l2_loss
            total_pearson += pearson
            total_delta_pearson += delta_pearson
            total_spearman += spearman
            total_delta_spearman += delta_spearman

            if batch % log_interval == 0 and model.local_rank == 0:
                avg_loss = total_loss / (batch + 1)
                avg_aux_loss = total_aux_loss / (batch + 1)
                avg_mse = total_mse / (batch + 1)
                avg_l2_loss = total_l2_loss / (batch + 1)
                avg_pearson = total_pearson / (batch + 1)
                avg_delta_pearson = total_delta_pearson / (batch + 1)
                avg_spearman = total_spearman / (batch + 1)
                avg_delta_spearman = total_delta_spearman / (batch + 1)
                print(
                    f"inf | Batch {batch}/{n_batch} | "
                    f"Avg total Loss: {avg_loss:.4f} | "
                    f"Avg aux Loss: {avg_aux_loss:.4f} | "
                    f"Avg mse: {avg_mse:.4f} | "
                    f"Avg l2 loss: {avg_l2_loss:.4f} | "
                    f"Avg pearson: {avg_pearson:.4f} | "
                    f"Avg delta pearson: {avg_delta_pearson:.4f} | "
                    f"Avg spearman: {avg_spearman:.4f} | "
                    f"Avg delta spearman: {avg_delta_spearman:.4f}"
                )

    avg_loss = total_loss / n_batch
    avg_mse = total_mse / n_batch
    avg_aux_loss = total_aux_loss / n_batch
    avg_l2_loss = total_l2_loss / n_batch
    avg_pearson = total_pearson / n_batch
    avg_delta_pearson = total_delta_pearson / n_batch
    avg_spearman = total_spearman / n_batch
    avg_delta_spearman = total_delta_spearman / n_batch
    pred_exp_all = torch.cat(pred_exp_list, dim=0)

    os.makedirs(save_dir, exist_ok=True)
    save_path_pred_exp = os.path.join(save_dir, "pred_exp.pt")
    torch.save(pred_exp_all, save_path_pred_exp)

    epoch_time = time.time() - start_time
    if model.local_rank == 0:
        print(
            f"inf completed in {epoch_time:.2f} seconds | "
            f"Avg total Loss: {avg_loss:.4f} | "
            f"Avg aux Loss: {avg_aux_loss:.4f} | "
            f"Avg mse: {avg_mse:.4f} | "
            f"Avg l2 loss: {avg_l2_loss:.4f} | "
            f"Avg pearson: {avg_pearson:.4f} | "
            f"Avg delta pearson: {avg_delta_pearson:.4f} | "
            f"Avg spearman: {avg_spearman:.4f} | "
            f"Avg delta spearman: {avg_delta_spearman:.4f}"
        )
        print(f"pred_exp_value saved to {save_path_pred_exp}")
