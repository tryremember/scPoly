import argparse
import os

import deepspeed
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader

from engine.checkpoint import load_safetensor_state_dict
from engine.collator import scCollatorPertGenetic
from engine.get_vocab import get_vocab
from engine.model import TransformerModel
from engine.run_epoch import inf_epoch


DATASET_CONFIGS = {
    "HEK293T": {
        "test_file": "../../data/xatlas/HEK293T/test_1700.parquet",
        "vocab_path": "../../data/xatlas/vocab_custom.json",
        "gene_order_path": "../../meta_info/gene_order.pkl",
    },
    "HCT116": {
        "test_file": "../../data/xatlas/HCT116/test_1700.parquet",
        "vocab_path": "../../data/xatlas/vocab_custom.json",
        "gene_order_path": "../../meta_info/gene_order.pkl",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for the unified perturbation genetic task.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Injected by deepspeed launcher.")
    parser.add_argument("--dataset", default="HEK293T", choices=sorted(DATASET_CONFIGS.keys()))
    parser.add_argument("--exp-id", required=True, help="Experiment directory name under save/<dataset>/.")
    parser.add_argument("--batch-size", type=int, default=8)
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


def main():
    args = parse_args()
    config = DATASET_CONFIGS[args.dataset]
    save_dir = os.path.join("./save", args.dataset, args.exp_id)
    checkpoint_path = os.path.join(save_dir, "model.safetensors")

    dataset = load_dataset("parquet", data_files=config["test_file"], split="train")
    collator = scCollatorPertGenetic(
        max_length=None,
        vocab_path=config["vocab_path"],
        gene_order_path=config["gene_order_path"],
        mask_exp=False,
        verbose=False,
        frozen_gene=True,
        inf=True,
    )
    test_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)

    vocab = get_vocab(config["vocab_path"])
    model = build_model(vocab)
    model.load_state_dict(load_safetensor_state_dict(checkpoint_path), strict=False)

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config="../../ckpt_scPoly/ds_config.json",
    )

    device = torch.device(f"cuda:{model_engine.local_rank}")
    os.makedirs(save_dir, exist_ok=True)
    inf_epoch(model_engine, test_loader, device, save_dir)


if __name__ == "__main__":
    main()



# nohup deepspeed --include localhost:0,1,2,3,4,5 --master_port 29632 infer.py --dataset HEK293T --exp-id 2026-06-22-12-00 > infer.log 2>&1 &
