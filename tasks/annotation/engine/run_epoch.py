import torch
import time
import os



def train_epoch(model, dataloader, device, optimizer, moe, epoch, log_interval, save_dir):
    """
    Run one training epoch.

    Iterates over the dataloader, performs forward and backward passes,
    updates model parameters, and logs running metrics. Supports both
    standard training and MoE-based training.

    Parameters
    ----------
    model : torch.nn.Module
        Training model.
    dataloader : DataLoader
        Training data loader.
    device : torch.device
        Target device.
    optimizer : torch.optim.Optimizer
        Optimizer used when moe=False.
    moe : bool
        Whether to use MoE backward/step logic.
    epoch : int
        Current epoch index.
    log_interval : int
        Logging interval in number of batches.
    save_dir : str
        Directory for optional outputs.
    """
    model.train()
    start_time = time.time()
    n_batch = len(dataloader)

    total_loss = 0.0
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_aux_loss = 0.0
    total_cls_loss = 0.0
    total_discriminator_loss = 0.0
    total_discriminator_acc = 0.0

    all_embeddings = []
    all_labels = []

    for batch, batch_data in enumerate(dataloader):
        batch_data = {k: v.to(model.device) for k, v in batch_data.items()}
        output = model(**batch_data)

        if "cell_emb" in output:
            all_embeddings.append(output["cell_emb"])
            all_labels.append(output["labels"])

        loss = output["total_loss"]
        accuracy = output["accuracy"]
        precision = output["precision"]
        recall = output["recall"]
        f1 = output["f1"]

        cls_loss = output["cls_loss"]
        aux_loss = output["l_aux"]

        discriminator_loss = None
        discriminator_acc = None
        if "discriminator_loss" in output and "discriminator_acc" in output:
            discriminator_loss = output["discriminator_loss"]
            discriminator_acc = output["discriminator_acc"]

        total_cls_loss += cls_loss.item()
        total_aux_loss += aux_loss
        total_loss += loss.item()
        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        if discriminator_loss is not None:
            total_discriminator_loss += discriminator_loss.item()
            total_discriminator_acc += discriminator_acc

        if moe:
            model.backward(loss)
            model.step()
        else:
            model.zero_grad()
            loss.backward()
            optimizer.step()

        if batch % log_interval == 0 and model.local_rank == 0:
            avg_loss = total_loss / (batch + 1)
            avg_accuracy = total_accuracy / (batch + 1)
            avg_precision = total_precision / (batch + 1)
            avg_recall = total_recall / (batch + 1)
            avg_f1 = total_f1 / (batch + 1)
            avg_cls_loss = total_cls_loss / (batch + 1)
            avg_aux_loss = total_aux_loss / (batch + 1)

            if discriminator_loss is not None:
                avg_discriminator_loss = total_discriminator_loss / (batch + 1)
                avg_discriminator_acc = total_discriminator_acc / (batch + 1)

            log_parts = [
                f"Epoch {epoch} | Batch {batch}/{n_batch}",
                f"Avg Loss: {avg_loss:.4f}",
                f"cls Loss: {avg_cls_loss:.4f}",
                f"aux Loss: {avg_aux_loss:.4f}",
                f"acc: {avg_accuracy:.4f}",
                f"precision: {avg_precision:.4f}",
                f"recall: {avg_recall:.4f}",
                f"f1: {avg_f1:.4f}",
                f"discriminator_loss: {avg_discriminator_loss:.4f}" if discriminator_loss is not None else "discriminator_loss: N/A",
                f"discriminator_acc: {avg_discriminator_acc:.4f}" if discriminator_acc is not None else "discriminator_acc: N/A",
            ]
            print(" | ".join(log_parts))

    epoch_time = time.time() - start_time
    if model.local_rank == 0:
        print(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")

    return


def valid_epoch(model, dataloader, device, epoch, save_dir):
    """
    Run one validation epoch.

    Evaluates the model on the validation set and reports averaged metrics.

    Parameters
    ----------
    model : torch.nn.Module
        Model in evaluation mode.
    dataloader : DataLoader
        Validation data loader.
    device : torch.device
        Target device.
    epoch : int
        Current epoch index.
    save_dir : str
        Directory for optional outputs.

    Returns
    -------
    float
        Average F1 score over the validation set.
    """
    model.eval()
    start_time = time.time()

    total_loss = 0.0
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_aux_loss = 0.0
    total_cls_loss = 0.0

    with torch.no_grad():
        for batch, batch_data in enumerate(dataloader):
            batch_data = {k: v.to(model.device) for k, v in batch_data.items()}
            output = model(inf=False, **batch_data)

            total_loss += output["total_loss"].item()
            total_accuracy += output["accuracy"]
            total_precision += output["precision"]
            total_recall += output["recall"]
            total_f1 += output["f1"]
            total_cls_loss += output["cls_loss"].item()
            total_aux_loss += output["l_aux"]

    n_batch = batch + 1
    avg_f1 = total_f1 / n_batch

    epoch_time = time.time() - start_time
    if model.local_rank == 0:
        print(f"Validation Epoch {epoch} completed in {epoch_time:.2f} seconds")

    return avg_f1



def inf_epoch(model, dataloader, device, save_dir):
    """
    Run inference over the dataset.

    Collects prediction logits, predicted labels, embeddings, and
    gating weights, and saves selected outputs to disk.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    dataloader : DataLoader
        Inference data loader.
    device : torch.device
        Target device.
    save_dir : str
        Output directory.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Concatenated prediction logits and predicted class indices.
    """
    model.eval()
    start_time = time.time()

    cls_pred_logits_list = []
    cls_pred_list = []
    cell_emb_list = []
    cell_emb_n_expert_list = []
    gate_weight_cls_list = []

    with torch.no_grad():
        for batch, batch_data in enumerate(dataloader):
            batch_data = {k: v.to(model.device) for k, v in batch_data.items()}
            output = model(inf=True, **batch_data)

            cls_pred_logits = output["logits"]
            cls_pred_logits_list.append(cls_pred_logits)

            cls_pred = torch.argmax(cls_pred_logits, dim=-1)
            cls_pred_list.append(cls_pred.cpu())

            cell_emb_list.append(output["cell_emb"])
            cell_emb_n_expert_list.append(output["cell_emb_n_expert"])
            gate_weight_cls_list.append(output["gate_weight_cls"])

    cls_pred_logits_all = torch.cat(cls_pred_logits_list, dim=0)
    cls_pred_all = torch.cat(cls_pred_list, dim=0)
    gate_weight_cls_all = torch.cat(gate_weight_cls_list, dim=0)

    os.makedirs(save_dir, exist_ok=True)

    torch.save(cls_pred_logits_all, os.path.join(save_dir, "cls_pred_logits.pt"))
    torch.save(cls_pred_all, os.path.join(save_dir, "cls_pred_intlabels.pt"))
    # torch.save(gate_weight_cls_all, os.path.join(save_dir, "gate_weight_cls.pt"))

    epoch_time = time.time() - start_time
    if model.local_rank == 0:
        print(f"Inference done in {epoch_time:.2f} seconds.")

    return cls_pred_logits_all, cls_pred_all
