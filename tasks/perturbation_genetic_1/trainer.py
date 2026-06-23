import argparse
import os
import time

import deepspeed
import torch
import torch.nn.functional as F
from datasets import load_dataset
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from engine.checkpoint import safe_load_model
from engine.collator import scCollatorPerturb
from engine.get_vocab import get_vocab
from engine.model import TransformerModel
from engine.run_epoch import train_epoch, valid_epoch


DATASET_CONFIGS = {
    "adamson": {
        "vocab_path": "../../data/adamson/vocab_custom.json",
        "gene_list_path": "../../data/adamson/gene_name_list.json",
        "train_file": "../../data/adamson/train.parquet",
        "val_file": "../../data/adamson/val.parquet",
        "gene_order_path": "../../meta_info/gene_order.pkl",
        "max_length": 1200,
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
        "nonzero_only_gene": True,
        "freeze_encoder": False,
    },
    "norman": {
        "vocab_path": "../../data/norman/vocab_custom.json",
        "gene_list_path": "../../data/norman/gene_name_list.json",
        "train_file": "../../data/norman/train.parquet",
        "val_file": "../../data/norman/val.parquet",
        "gene_order_path": "../../meta_info/gene_order.pkl",
        "max_length": 1200,
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
        "nonzero_only_gene": True,
        "freeze_encoder": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train the unified perturbation task.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Injected by deepspeed launcher.")
    parser.add_argument("--dataset", default="adamson", choices=sorted(DATASET_CONFIGS.keys()))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--freeze-encoder", action="store_true")
    return parser.parse_args()


def build_model(config, vocab):
    return TransformerModel(
        ntokens=len(vocab),
        num_layers=config["num_layers"],
        d_model=config["embsize"],
        nhead=config["nhead"],
        n_cls=None,
        pad_token="<pad>",
        vocab=vocab,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=True,
        norm_first=False,
        bias=True,
        device=None,
        dtype=None,
        moe=True,
        mlm=False,
        cls=False,
        pert=True,
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
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"{name} is frozen")


def train_model(model, optimizer, train_loader, val_loader, num_epochs, device, log_interval, save_dir):
    best_val_score = float("inf")

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
        val_score = valid_epoch(
            model,
            dataloader=val_loader,
            device=device,
            epoch=epoch,
            log_interval=log_interval,
            save_dir=save_dir,
        )
        if val_score < best_val_score and model.local_rank == 0:
            best_val_score = val_score
            model_to_save = model.module if hasattr(model, "module") else model
            model_path = os.path.join(save_dir, "model.safetensors")
            save_file(model_to_save.state_dict(), model_path)
            print(f" New best model saved at epoch {epoch} with val score: {val_score:.4f}")

        if model.local_rank == 0:
            print(f"========================== Epoch {epoch}/{num_epochs} end ============================\n\n")

    if model.local_rank == 0:
        print(f"{num_epochs} epochs done!\n")
        print(f"Best model saved in {save_dir} with score: {best_val_score:.4f}")


def main():
    args = parse_args()
    config = DATASET_CONFIGS[args.dataset].copy()

    ds_config_path = "../../ckpt_scPoly/ds_config.json"
    checkpoint_path = "../../ckpt_scPoly/model.safetensors"
    ckpt_save_root = "./save"

    batch_size = args.batch_size or config["batch_size"]
    num_epochs = args.num_epochs or config["num_epochs"]
    freeze_encoder = args.freeze_encoder or config["freeze_encoder"]

    current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    save_dir = os.path.join(ckpt_save_root, args.dataset, current_time)
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = load_dataset("parquet", data_files=config["train_file"], split="train")
    val_dataset = load_dataset("parquet", data_files=config["val_file"], split="train")

    collator_train = scCollatorPerturb(
        max_length=config["max_length"],
        vocab_path=config["vocab_path"],
        gene_order_path=config["gene_order_path"],
        gene_list_path=config["gene_list_path"],
        mask_exp=False,
        verbose=False,
        inf=False,
        nonzero_only_gene=config["nonzero_only_gene"],
    )
    collator_val = scCollatorPerturb(
        max_length=config["max_length"],
        vocab_path=config["vocab_path"],
        gene_order_path=config["gene_order_path"],
        gene_list_path=config["gene_list_path"],
        mask_exp=False,
        verbose=False,
        inf=True,
        nonzero_only_gene=config["nonzero_only_gene"],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collator_val)

    vocab = get_vocab(config["vocab_path"])
    model = build_model(config, vocab)
    model = safe_load_model(
        model,
        checkpoint_path,
        ignore_keys=[],
        partial_load={"encoder1.embedding.weight": None},
        verbose=True,
    )

    if freeze_encoder:
        for name, param in model.named_parameters():
            if "enc" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable_params = model.parameters()

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=trainable_params,
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



# nohup deepspeed --include localhost:0,1,2,3,4,5 --master_port 29621 trainer.py --dataset adamson > trainer.log 2>&1 &
