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
    total_mse_loss = 0.0
    total_mse_loss_bycls = 0.0

    for batch, batch_data in enumerate(dataloader):
        batch_data = {k: v.to(model.device) for k, v in batch_data.items()}
        output = model(**batch_data)

        loss = output["total_loss"]
        aux_loss = output["l_aux"]
        mse_loss = output["mse_loss"]
        mse_loss_bycls = output["mse_loss_bycls"]

        total_aux_loss += aux_loss
        total_loss += loss.item()
        total_mse_loss += mse_loss.item()
        total_mse_loss_bycls += mse_loss_bycls.item()

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
            avg_mse_loss = total_mse_loss / (batch + 1)
            avg_mse_loss_bycls = total_mse_loss_bycls / (batch + 1)
            print(
                f"Epoch {epoch} | Batch {batch}/{n_batch} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"aux Loss: {avg_aux_loss:.4f} | "
                f"mse Loss: {avg_mse_loss:.4f} | "
                f"mse Loss by cls: {avg_mse_loss_bycls:.4f}"
            )

    avg_loss = total_loss / n_batch
    avg_aux_loss = total_aux_loss / n_batch
    avg_mse_loss = total_mse_loss / n_batch
    avg_mse_loss_bycls = total_mse_loss_bycls / n_batch
    epoch_time = time.time() - start_time

    if model.local_rank == 0:
        print(
            f"Epoch {epoch} completed in {epoch_time:.2f} seconds | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"aux Loss: {avg_aux_loss:.4f} | "
            f"mse Loss: {avg_mse_loss:.4f} | "
            f"mse Loss by cls: {avg_mse_loss_bycls:.4f}"
        )


def valid_epoch(model, dataloader, device, epoch, log_interval, save_dir):
    del device, save_dir, log_interval
    model.eval()
    start_time = time.time()
    total_loss = 0.0
    total_aux_loss = 0.0
    total_mse_loss = 0.0
    total_mse_loss_bycls = 0.0

    with torch.no_grad():
        for batch, batch_data in enumerate(dataloader):
            batch_data = {k: v.to(model.device) for k, v in batch_data.items()}
            output = model(inf=False, **batch_data)
            total_loss += output["total_loss"].item()
            total_aux_loss += output["l_aux"]
            total_mse_loss += output["mse_loss"].item()
            total_mse_loss_bycls += output["mse_loss_bycls"].item()

    n_batch = batch + 1
    avg_loss = total_loss / n_batch
    avg_aux_loss = total_aux_loss / n_batch
    avg_mse_loss = total_mse_loss / n_batch
    avg_mse_loss_bycls = total_mse_loss_bycls / n_batch
    epoch_time = time.time() - start_time

    if model.local_rank == 0:
        print(
            f"Validation Epoch {epoch} completed in {epoch_time:.2f} seconds | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"aux Loss: {avg_aux_loss:.4f} | "
            f"mse Loss: {avg_mse_loss:.4f} | "
            f"mse Loss by cls: {avg_mse_loss_bycls:.4f}"
        )

    return avg_loss


def inf_epoch(model, dataloader, device, save_dir, tag):
    del device
    model.eval()
    start_time = time.time()
    total_aux_loss = 0.0
    attn_weights_mean = None

    with torch.no_grad():
        for batch, batch_data in enumerate(dataloader):
            batch_data = {k: v.to(model.device) for k, v in batch_data.items()}
            output = model(inf=True, **batch_data)
            total_aux_loss += output["l_aux"]
            attn_weights_batch = output["attn_weights_celltype"]

            if attn_weights_mean is None:
                attn_weights_mean = attn_weights_batch
            else:
                attn_weights_mean = (attn_weights_mean * batch + attn_weights_batch) / (batch + 1)

            if model.local_rank == 0:
                print(f"aux Loss: {output['l_aux']:.4f}")

    avg_aux_loss = total_aux_loss / (batch + 1)
    os.makedirs(save_dir, exist_ok=True)
    save_path_attn = os.path.join(save_dir, f"attn_weights_{tag}.pt")
    torch.save(attn_weights_mean, save_path_attn)

    epoch_time = time.time() - start_time
    if model.local_rank == 0:
        print(f"Inference done in {epoch_time:.2f} seconds.")
        print(f"aux Loss: {avg_aux_loss:.4f}")
        print(f"Attention weights saved to {save_path_attn}")

    return attn_weights_mean
