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
from engine.collator import scCollatorPertGenetic
from engine.get_vocab import get_vocab
from engine.model import TransformerModel
from engine.run_epoch import train_epoch, valid_epoch


DATASET_CONFIGS = {
    "HEK293T": {
        "train_file": "../../data/xatlas/HEK293T/train_1700.parquet",
        "vocab_path": "../../data/xatlas/vocab_custom.json",
        "gene_order_path": "../../meta_info/gene_order.pkl",
    },
    "HCT116": {
        "train_file": "../../data/xatlas/HCT116/train_1700.parquet",
        "vocab_path": "../../data/xatlas/vocab_custom.json",
        "gene_order_path": "../../meta_info/gene_order.pkl",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train the unified perturbation genetic task.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Injected by deepspeed launcher.")
    parser.add_argument("--dataset", default="HEK293T", choices=sorted(DATASET_CONFIGS.keys()))
    parser.add_argument("--checkpoint-path", default="/home/24144449r/desktop/MoE/pretrain_v3/save_0522/checkpoint-440000/model.safetensors")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--freeze-encoder", action="store_true")
    return parser.parse_args()


def build_model(vocab):
    return TransformerModel(
        num_layers=12,
        ntokens=len(vocab),
        d_model=512,
        nhead=8,
        seq_len=None,
        n_cls=None,
        n_batch=None,
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
        GRL=False,
        mlm=False,
        CL=False,
        cls=False,
        pert=True,
        dim_feedforward=1024,
        alpha=1,
        beta=1,
        moe_experts=6,
        epsize=1,
        top_k=2,
        moe_layers=[3, 5, 7, 9, 11],
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
        train_epoch(model, dataloader=train_loader, device=device, optimizer=optimizer, moe=True, epoch=epoch, log_interval=log_interval, save_dir=save_dir)
        val_score = valid_epoch(model, dataloader=val_loader, device=device, epoch=epoch, save_dir=save_dir)
        if val_score < best_val_score and model.local_rank == 0:
            best_val_score = val_score
            model_to_save = model.module if hasattr(model, "module") else model
            model_path = os.path.join(save_dir, "model.safetensors")
            save_file(model_to_save.state_dict(), model_path)
            if model.local_rank == 0:
                print(f" New best model saved at epoch {epoch} with val score: {val_score:.4f}")

        if model.local_rank == 0:
            print(f"========================== Epoch {epoch}/{num_epochs} end ============================\n\n")

    if model.local_rank == 0:
        print(f"{num_epochs} epochs done!\n")
        print(f"Best model saved in {save_dir} with score: {best_val_score:.4f}")


def main():
    args = parse_args()
    config = DATASET_CONFIGS[args.dataset]

    train_path = config["train_file"]
    dataset = load_dataset("parquet", data_files=train_path, split="train")
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]

    collator = scCollatorPertGenetic(
        max_length=None,
        vocab_path=config["vocab_path"],
        gene_order_path=config["gene_order_path"],
        mask_exp=False,
        verbose=False,
        frozen_gene=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collator)

    vocab = get_vocab(config["vocab_path"])
    model = build_model(vocab)
    model = safe_load_model(
        model,
        args.checkpoint_path,
        ignore_keys=[".gate."],
        partial_load={},
        verbose=True,
    )

    if args.freeze_encoder:
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
        config="../../ckpt_scPoly/ds_config.json",
    )

    parameter_stat(model)

    current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    save_dir = os.path.join("./save", args.dataset, current_time)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(f"cuda:{model_engine.local_rank}")
    print(f"experiment -- {args.dataset} -- {current_time}")
    train_model(model_engine, optimizer, train_loader, val_loader, args.num_epochs, device, 100, save_dir)


if __name__ == "__main__":
    main()



# nohup deepspeed --include localhost:0,1,2,3,4,5 --master_port 29631 trainer.py --dataset HEK293T > trainer.log 2>&1 &
