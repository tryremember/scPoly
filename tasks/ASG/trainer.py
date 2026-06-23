import argparse
import os
import time

import deepspeed
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from engine.checkpoint import safe_load_model
from engine.collator import scCollatorASG
from engine.data_utils import h5ad2dataset, preprocess_h5ad
from engine.get_vocab import get_vocab
from engine.model import TransformerModel
from engine.run_epoch import train_epoch, valid_epoch


DATASET_CONFIGS = {
    "ms_23en_ctrl_10k": {
        "h5ad_path": "../../data/ms/23en/23en_ctrl_oversample10k.h5ad",
        "vocab_path": "../../data/ms/23en/vocab_custom.json",
        "gene_order_path": "../../meta_info/gene_order.pkl",
        "checkpoint_path": "../../ckpt_scPoly/model.safetensors",
        "max_length": None,
        "num_layers": 12,
        "embsize": 512,
        "nhead": 1,
        "dim_feedforward": 1024,
        "moe_experts": 6,
        "epsize": 1,
        "top_k": 1,
        "moe_layers": [3, 5, 7, 9, 11],
        "batch_size": 8,
        "num_epochs": 50,
        "log_interval": 100,
        "val_ratio": 0.1,
        "do_norm_and_log1p": False,
        "frozen_gene": True,
        "mask_ratio": 0.15,
    },
    "ms_23en_ms_10k": {
        "h5ad_path": "../../data/ms/23en/23en_ms_oversample10k.h5ad",
        "vocab_path": "../../data/ms/23en/vocab_custom.json",
        "gene_order_path": "../../meta_info/gene_order.pkl",
        "checkpoint_path": "../../ckpt_scPoly/model.safetensors",
        "max_length": None,
        "num_layers": 12,
        "embsize": 512,
        "nhead": 1,
        "dim_feedforward": 1024,
        "moe_experts": 6,
        "epsize": 1,
        "top_k": 1,
        "moe_layers": [3, 5, 7, 9, 11],
        "batch_size": 8,
        "num_epochs": 50,
        "log_interval": 100,
        "val_ratio": 0.1,
        "do_norm_and_log1p": False,
        "frozen_gene": True,
        "mask_ratio": 0.15,
    },
    "ms_opc_ctrl_10k": {
        "h5ad_path": "../../data/ms/opc/opc_ctrl_oversample10k.h5ad",
        "vocab_path": "../../data/ms/opc/vocab_custom.json",
        "gene_order_path": "../../meta_info/gene_order.pkl",
        "checkpoint_path": "../../ckpt_scPoly/model.safetensors",
        "max_length": None,
        "num_layers": 12,
        "embsize": 512,
        "nhead": 1,
        "dim_feedforward": 1024,
        "moe_experts": 6,
        "epsize": 1,
        "top_k": 1,
        "moe_layers": [3, 5, 7, 9, 11],
        "batch_size": 8,
        "num_epochs": 50,
        "log_interval": 100,
        "val_ratio": 0.1,
        "do_norm_and_log1p": False,
        "frozen_gene": True,
        "mask_ratio": 0.15,
    },
    "ms_opc_ms_10k": {
        "h5ad_path": "../../data/ms/opc/opc_ms_oversample10k.h5ad",
        "vocab_path": "../../data/ms/opc/vocab_custom.json",
        "gene_order_path": "../../meta_info/gene_order.pkl",
        "checkpoint_path": "../../ckpt_scPoly/model.safetensors",
        "max_length": None,
        "num_layers": 12,
        "embsize": 512,
        "nhead": 1,
        "dim_feedforward": 1024,
        "moe_experts": 6,
        "epsize": 1,
        "top_k": 1,
        "moe_layers": [3, 5, 7, 9, 11],
        "batch_size": 8,
        "num_epochs": 50,
        "log_interval": 100,
        "val_ratio": 0.1,
        "do_norm_and_log1p": False,
        "frozen_gene": True,
        "mask_ratio": 0.15,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train the unified ASG task.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Injected by deepspeed launcher.")
    parser.add_argument("--dataset", default="ms_23en_ctrl_10k", choices=sorted(DATASET_CONFIGS.keys()))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    return parser.parse_args()


def build_model(config, vocab, mlm=True):
    return TransformerModel(
        ntokens=len(vocab),
        num_layers=config["num_layers"],
        d_model=config["embsize"],
        nhead=config["nhead"],
        seq_len=config["max_length"],
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
        mlm=mlm,
        dim_feedforward=config["dim_feedforward"],
        moe_experts=config["moe_experts"],
        epsize=config["epsize"],
        top_k=config["top_k"],
        moe_layers=config["moe_layers"],
    )


def parameter_stat(model):
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_trainable}")
    num_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Frozen parameters: {num_frozen}")


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
        print(f"{num_epochs} epochs done!")
        print(f"Best model saved in {save_dir} with score: {best_val_score:.4f}")


def main():
    args = parse_args()
    config = DATASET_CONFIGS[args.dataset].copy()

    ds_config_path = "../../ckpt_scPoly/ds_config.json"
    ckpt_save_root = "./save"
    batch_size = args.batch_size or config["batch_size"]
    num_epochs = args.num_epochs or config["num_epochs"]

    current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    save_dir = os.path.join(ckpt_save_root, args.dataset, current_time)
    os.makedirs(save_dir, exist_ok=True)

    adata = preprocess_h5ad(config["h5ad_path"], do_norm_and_log1p=config["do_norm_and_log1p"])
    dataset, _, _ = h5ad2dataset(adata, config["vocab_path"], frozen_gene=config["frozen_gene"], inf=False)
    split = dataset.train_test_split(test_size=config["val_ratio"], seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]

    collator = scCollatorASG(
        max_length=config["max_length"],
        vocab_path=config["vocab_path"],
        gene_order_path=config["gene_order_path"],
        mask_ratio=config["mask_ratio"],
        mask_exp=True,
        verbose=False,
        inf=False,
        frozen_gene=config["frozen_gene"],
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    vocab = get_vocab(config["vocab_path"])
    model = build_model(config, vocab, mlm=True)
    model = safe_load_model(
        model,
        config["checkpoint_path"],
        ignore_keys=[".gate."],
        partial_load={},
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



# nohup deepspeed --include localhost:5 --master_port 29641 trainer.py --dataset ms_23en_ctrl_10k > train.log 2>&1 &
