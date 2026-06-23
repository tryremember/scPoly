import torch
import time
import os



# writer
def train_epoch(model, dataloader, device, optimizer, moe, epoch,log_interval,save_dir):
    model.train()
    start_time = time.time()
    n_batch = len(dataloader)

    total_loss = 0.0
    total_aux_loss = 0.0
    total_mse_loss = 0.0
    total_mse_loss_bycls = 0.0
    total_contrastive_loss = 0.0
    total_discriminator_loss = 0.0
    total_discriminator_acc = 0.0
    total_pearson = 0.0
    total_delta_pearson = 0.0


    all_embeddings = []
    all_labels = []

    for batch, batch_data in enumerate(dataloader):
        batch_data = {k: v.to(model.device) for k, v in batch_data.items()}
        output = model(**batch_data) 



        loss = output['total_loss']

        aux_loss = output['l_aux']
        mse_loss = output['mse_loss']
        mse_loss_bycls = output['mse_loss_bycls']
        pearson = output['pearson']
        delta_pearson = output['delta_pearson']

        contrastive_loss = None
        discriminator_loss = None
        discriminator_acc = None

        if "discriminator_loss" in output and "discriminator_acc" in output:
            discriminator_loss = output['discriminator_loss']
            discriminator_acc = output['discriminator_acc']
        if "contrastive_loss" in output:
            contrastive_loss = output["contrastive_loss"]


        total_aux_loss += aux_loss
        total_loss += loss.item()
        total_mse_loss += mse_loss.item()
        total_mse_loss_bycls += mse_loss_bycls.item()
        total_pearson += pearson.item()
        total_delta_pearson += delta_pearson.item()

        if "discriminator_loss" in output and "discriminator_acc" in output:
            total_discriminator_loss += discriminator_loss.item()
            total_discriminator_acc += discriminator_acc
        if "contrastive_loss" in output:
            total_contrastive_loss += contrastive_loss.item()


        # Using MOE
        if moe:
            model.backward(loss)
            model.step()
        else:
            model.zero_grad()
            loss.backward()
            optimizer.step()

        # Print average metrics every log_interval batches (100 here)
        if batch % log_interval == 0 and model.local_rank == 0:
            avg_loss = total_loss / (batch + 1)

            avg_aux_loss = total_aux_loss / (batch+1)
            avg_mse_loss = total_mse_loss / (batch+1)
            avg_mse_loss_bycls = total_mse_loss_bycls / (batch+1)
            avg_pearson = total_pearson / (batch+1)
            avg_delta_pearson = total_delta_pearson / (batch+1)




            if "discriminator_loss" in output and "discriminator_acc" in output:
                avg_discriminator_loss = total_discriminator_loss/ (batch+1)
                avg_discriminator_acc = total_discriminator_acc/ (batch+1)
            if "contrastive_loss" in output:
                avg_contrastive_loss = total_contrastive_loss/ (batch+1)



            
            parts = [
                f"Epoch {epoch} | Batch {batch}/{n_batch}",
                f"Avg Loss: {avg_loss:.4f}",
                f"aux Loss: {avg_aux_loss:.4f}",
                f"mse Loss: {avg_mse_loss:.4f}",
                f"mse Loss by cls: {avg_mse_loss_bycls:.4f}",
                f"pearson: {avg_pearson:.4f}",
                f"delta pearson: {avg_delta_pearson:.4f}",
                f"contrastive_loss:{avg_contrastive_loss:.4f}" if contrastive_loss is not None else "contrastive_loss:N/A",
                f"discriminator_loss:{avg_discriminator_loss:.4f}" if discriminator_loss is not None else "discriminator_loss: N/A",
                f"discriminator_acc:{avg_discriminator_acc:.4f}" if discriminator_acc is not None else "discriminator_acc: N/A",
            ]
            print(" | ".join(parts))



    avg_loss = total_loss / n_batch
    avg_mse_loss = total_mse_loss / n_batch
    avg_mse_loss_bycls = total_mse_loss_bycls / n_batch
    avg_pearson = total_pearson / n_batch
    avg_delta_pearson = total_delta_pearson / n_batch

    if "discriminator_loss" in output and "discriminator_acc" in output:
        avg_discriminator_acc = total_discriminator_acc / n_batch
        avg_discriminator_loss = total_discriminator_loss / n_batch
    if "contrastive_loss" in output:
        avg_contrastive_loss = total_contrastive_loss / n_batch
        

    epoch_time = time.time() - start_time
    if model.local_rank == 0:

        parts = [
            f"Epoch {epoch} completed in {epoch_time:.2f} seconds",
            f"Epoch {epoch} | Batch {batch}/{n_batch}",
            f"Avg Loss: {avg_loss:.4f}",
            f"aux Loss: {avg_aux_loss:.4f}",
            f"mse Loss: {avg_mse_loss:.4f}",
            f"mse Loss by cls: {avg_mse_loss_bycls:.4f}",
            f"pearson: {avg_pearson:.4f}",
            f"delta pearson: {avg_delta_pearson:.4f}",
            f"contrastive_loss:{avg_contrastive_loss:.4f}" if contrastive_loss is not None else "contrastive_loss:N/A",
            f"discriminator_loss:{avg_discriminator_loss:.4f}" if discriminator_loss is not None else "discriminator_loss: N/A",
            f"discriminator_acc:{avg_discriminator_acc:.4f}" if discriminator_acc is not None else "discriminator_acc: N/A",
        ]
        print(" | ".join(parts))

    return


def valid_epoch(model, dataloader, device, epoch, save_dir):
    model.eval()
    start_time = time.time()
    n_batch = len(dataloader)

    total_loss = 0.0
    total_aux_loss = 0.0
    total_mse_loss = 0.0
    total_mse_loss_bycls = 0.0
    total_contrastive_loss = 0.0
    total_discriminator_loss = 0.0
    total_discriminator_acc = 0.0
    total_pearson = 0.0
    total_delta_pearson = 0.0

    with torch.no_grad():
        for batch, batch_data in enumerate(dataloader):
            batch_data = {k: v.to(model.device) for k, v in batch_data.items()}
            output = model(inf=False, **batch_data)

            loss = output["total_loss"]
            mse_loss = output["mse_loss"]
            aux_loss = output["l_aux"]
            mse_loss_bycls = output["mse_loss_bycls"]
            pearson = output["pearson"]
            delta_pearson = output["delta_pearson"]

            contrastive_loss = None
            discriminator_loss = None
            discriminator_acc = None

            if "discriminator_loss" in output and "discriminator_acc" in output:
                discriminator_loss = output["discriminator_loss"]
                discriminator_acc = output["discriminator_acc"]
            if "contrastive_loss" in output:
                contrastive_loss = output["contrastive_loss"]

            total_aux_loss += aux_loss
            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_mse_loss_bycls += mse_loss_bycls.item()
            total_pearson += pearson.item()
            total_delta_pearson += delta_pearson.item()

            if "discriminator_loss" in output and "discriminator_acc" in output:
                total_discriminator_loss += discriminator_loss.item()
                total_discriminator_acc += discriminator_acc
            if "contrastive_loss" in output:
                total_contrastive_loss += contrastive_loss.item()

    avg_loss = total_loss / (batch + 1)
    avg_aux_loss = total_aux_loss / (batch + 1)
    avg_mse_loss = total_mse_loss / (batch + 1)
    avg_mse_loss_bycls = total_mse_loss_bycls / (batch + 1)
    avg_pearson = total_pearson / (batch + 1)
    avg_delta_pearson = total_delta_pearson / (batch + 1)

    if "discriminator_loss" in output and "discriminator_acc" in output:
        avg_discriminator_loss = total_discriminator_loss / (batch + 1)
        avg_discriminator_acc = total_discriminator_acc / (batch + 1)
    if "contrastive_loss" in output:
        avg_contrastive_loss = total_contrastive_loss / (batch + 1)

    epoch_time = time.time() - start_time
    if model.local_rank == 0:
        parts = [
            f"Validation Epoch {epoch} completed in {epoch_time:.2f} seconds",
            f"Avg Loss: {avg_loss:.4f}",
            f"aux Loss: {avg_aux_loss:.4f}",
            f"mse Loss: {avg_mse_loss:.4f}",
            f"mse Loss by cls: {avg_mse_loss_bycls:.4f}",
            f"pearson: {avg_pearson:.4f}",
            f"delta pearson: {avg_delta_pearson:.4f}",
            f"contrastive_loss:{avg_contrastive_loss:.4f}" if contrastive_loss is not None else "contrastive_loss:N/A",
            f"discriminator_loss:{avg_discriminator_loss:.4f}" if discriminator_loss is not None else "discriminator_loss: N/A",
            f"discriminator_acc:{avg_discriminator_acc:.4f}" if discriminator_acc is not None else "discriminator_acc: N/A",
        ]
        print(" | ".join(parts))

    return avg_loss


def inf_epoch(model, dataloader, device, save_dir, which_celltype=None, which_ggi_index=None):
    model.eval()
    start_time = time.time()
    total_aux_loss = 0.0
    pred_exp_value_bygene_list = []

    with torch.no_grad():
        for batch, batch_data in enumerate(dataloader):
            batch_data = {k: v.to(model.device) for k, v in batch_data.items()}
            output = model(inf=True, **batch_data)

            aux_loss = output["l_aux"]
            total_aux_loss += aux_loss
            pred_exp_value_bygene = output["pred_exp_value"]
            pred_exp_value_bygene_list.append(pred_exp_value_bygene)

            if model.local_rank == 0:
                print(f" aux Loss: {aux_loss:.4f} | ")

    avg_aux_loss = total_aux_loss / (batch + 1)
    pred_exp_value_bygene_all = torch.cat(pred_exp_value_bygene_list, dim=0)

    os.makedirs(save_dir, exist_ok=True)
    save_path_pred_exp_value_bygene = os.path.join(save_dir, "pred_exp_value.pt")
    torch.save(pred_exp_value_bygene_all, save_path_pred_exp_value_bygene)

    epoch_time = time.time() - start_time
    if model.local_rank == 0:
        print(f"Inference done in {epoch_time} seconds.")
        print(f" aux Loss: {avg_aux_loss:.4f} | ")

    return pred_exp_value_bygene_all

