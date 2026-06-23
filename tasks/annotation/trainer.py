import argparse
import json
import os
import time

import deepspeed
import torch
import torch.nn.functional as F
from datasets import Dataset
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from engine.checkpoint import safe_load_model
from engine.collator import scCollator
from engine.data_utils import preprocess_h5ad
from engine.get_vocab import get_vocab
from engine.model import TransformerModel
from engine.run_epoch import train_epoch, valid_epoch
from engine.tokenizer import h5ad2parquet


DATASET_CONFIGS = {
    "pancreas16k": {
        "train_file": "train0.05.h5ad",
        "vocab_path": "../../meta_info/vocab.json",
        "gene_order_path": "../../meta_info/gene_order.pkl",
        "max_length": 1201,
        "num_layers": 12,
        "embsize": 512,
        "nhead": 8,
        "dim_feedforward": 2048,
        "moe_experts": 6,
        "epsize": 1,
        "top_k": 2,
        "batch_size": 8,
        "num_epochs": 5,
        "log_interval": 100,
        "do_norm_and_log1p": False,
    },
    "COAD": {
        "train_file": "train.h5ad",
        "vocab_path": "../../meta_info/vocab.json",
        "gene_order_path": "../../meta_info/gene_order.pkl",
        "max_length": 1201,
        "num_layers": 12,
        "embsize": 512,
        "nhead": 8,
        "dim_feedforward": 2048,
        "moe_experts": 6,
        "epsize": 1,
        "top_k": 2,
        "batch_size": 8,
        "num_epochs": 5,
        "log_interval": 100,
        "do_norm_and_log1p": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train the unified annotation task.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Injected by deepspeed launcher.")
    parser.add_argument(
        "--dataset",
        default="pancreas16k",
        choices=sorted(DATASET_CONFIGS.keys()),
        help="Dataset preset to use.",
    )
    parser.add_argument(
        "--h5ad-path",
        default=None,
        help="Optional override for the training h5ad file.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override epoch count.")
    return parser.parse_args()


def build_model(config, vocab, num_cls, num_batch):
    return TransformerModel(
        ntokens=len(vocab),
        num_layers=config["num_layers"],
        d_model=config["embsize"],
        nhead=config["nhead"],
        n_cls=num_cls,
        n_batch=num_batch,
        pad_token="<pad>",
        vocab=vocab,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=True,
        bias=True,
        device=None,
        dtype=None,
        GRL=False,
        mlm=False,
        cls=True,
        alpha=None,
        dim_feedforward=config["dim_feedforward"],
        moe_experts=config["moe_experts"],
        epsize=config["epsize"],
        top_k=config["top_k"],
    )


def parameter_stat(model):
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_trainable}")
    num_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Frozen parameters: {num_frozen}")


def train_model(model, optimizer, train_loader, val_loader, num_epochs, device, log_interval, save_dir):
    best_val_score = float("-inf")

    for epoch in range(1, num_epochs + 1):
        if model.local_rank == 0:
            print(f"========================== Epoch {epoch}/{num_epochs} begin ==========================")

        train_epoch(
            model,
            dataloader=train_loader,
            device=device,
            optimizer=optimizer,
            moe=True,
            epoch=epoch,
            log_interval=log_interval,
            save_dir=save_dir,
        )
        val_score = valid_epoch(model, dataloader=val_loader, device=device, epoch=epoch, save_dir=save_dir)

        if val_score > best_val_score and model.local_rank == 0:
            best_val_score = val_score
            model_to_save = model.module if hasattr(model, "module") else model
            model_path = os.path.join(save_dir, "model.safetensors")
            save_file(model_to_save.state_dict(), model_path)
            if model.local_rank == 0:
                print(f"New best model saved at epoch {epoch} with val score: {val_score:.4f}")

        if model.local_rank == 0:
            print(f"========================== Epoch {epoch}/{num_epochs} end ============================\n")

    if model.local_rank == 0:
        print(f"Best model saved in {save_dir} with score: {best_val_score:.4f}")


def main():
    args = parse_args()
    config = DATASET_CONFIGS[args.dataset].copy()

    ds_config_path = "../../ckpt_scPoly/ds_config.json"
    checkpoint_path = "../../ckpt_scPoly/model.safetensors"
    ckpt_save_root = "./save"
    data_root = "../../data"

    batch_size = args.batch_size or config["batch_size"]
    num_epochs = args.num_epochs or config["num_epochs"]

    current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    save_dir = os.path.join(ckpt_save_root, args.dataset, current_time)
    os.makedirs(save_dir, exist_ok=True)

    h5ad_path = args.h5ad_path or os.path.join(data_root, args.dataset, config["train_file"])
    adata = preprocess_h5ad(h5ad_path, do_norm_and_log1p=config["do_norm_and_log1p"])
    tokenized_data, num_cls, num_batch, celltype2int, _, batch2int, _, _ = h5ad2parquet(
        adata,
        config["vocab_path"],
        celltype_key="celltype",
        batch_key="batch",
    )
    dataset = Dataset.from_list(tokenized_data)

    with open(os.path.join(save_dir, "celltype2int.json"), "w") as f:
        json.dump(celltype2int, f)
    with open(os.path.join(save_dir, "batch2int.json"), "w") as f:
        json.dump(batch2int, f)

    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]

    collator = scCollator(
        max_length=config["max_length"],
        vocab_path=config["vocab_path"],
        gene_order_path=config["gene_order_path"],
        mask_exp=False,
        verbose=False,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collator)

    vocab = get_vocab(config["vocab_path"])
    model = build_model(config, vocab, num_cls, num_batch)
    model = safe_load_model(
        model,
        checkpoint_path,
        ignore_keys=[],
        partial_load={"encoder1.embedding.weight": None},
        verbose=True,
    )

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config_path,
    )

    parameter_stat(model)

    device = torch.device(f"cuda:{model_engine.local_rank}")
    print(f"experiment -- {args.dataset} -- {current_time}")
    train_model(
        model_engine,
        optimizer,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        device=device,
        log_interval=config["log_interval"],
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()



# nohup deepspeed --include localhost:0,1,2,3,4,5 --master_port 29611 trainer.py --dataset pancreas16k > trainer.log 2>&1 &