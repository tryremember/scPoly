import argparse
import json
import os

import deepspeed
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader

from engine.checkpoint import load_safetensor_state_dict
from engine.collator import scCollator
from engine.data_utils import preprocess_h5ad
from engine.get_vocab import get_vocab
from engine.model import TransformerModel
from engine.run_epoch import inf_epoch
from engine.tokenizer import h5ad2parquet


DATASET_CONFIGS = {
    "pancreas16k": {
        "infer_file": "test.h5ad",
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
        "do_norm_and_log1p": False,
    },
    "COAD": {
        "infer_file": "test_xenium.h5ad",
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
        "do_norm_and_log1p": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for the unified annotation task.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Injected by deepspeed launcher.")
    parser.add_argument(
        "--dataset",
        default="pancreas16k",
        choices=sorted(DATASET_CONFIGS.keys()),
        help="Dataset preset to use.",
    )
    parser.add_argument("--exp-id", required=True, help="Experiment directory name under save/<dataset>/.")
    parser.add_argument("--h5ad-path", default=None, help="Optional override for the inference h5ad file.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    return parser.parse_args()


def build_model(config, vocab, num_cls):
    return TransformerModel(
        ntokens=len(vocab),
        num_layers=config["num_layers"],
        d_model=config["embsize"],
        nhead=config["nhead"],
        n_cls=num_cls,
        n_batch=None,
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


def main():
    args = parse_args()
    config = DATASET_CONFIGS[args.dataset]

    ds_config_path = "../../ckpt_scPoly/ds_config.json"
    data_root = "../../data"
    save_dir = os.path.join("./save", args.dataset, args.exp_id)
    checkpoint_path = os.path.join(save_dir, "model.safetensors")
    mapping_path = os.path.join(save_dir, "celltype2int.json")

    h5ad_path = args.h5ad_path or os.path.join(data_root, args.dataset, config["infer_file"])
    batch_size = args.batch_size or config["batch_size"]

    adata = preprocess_h5ad(h5ad_path, do_norm_and_log1p=config["do_norm_and_log1p"])
    tokenized_data = h5ad2parquet(adata, config["vocab_path"], inf=True)
    dataset = Dataset.from_list(tokenized_data)

    with open(mapping_path, "r") as f:
        celltype2int = json.load(f)
    num_cls = len(celltype2int)
    print(f"Number of classes: {num_cls}")

    collator = scCollator(
        max_length=config["max_length"],
        vocab_path=config["vocab_path"],
        gene_order_path=config["gene_order_path"],
        mask_exp=False,
        verbose=False,
        inf=True,
    )
    test_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    vocab = get_vocab(config["vocab_path"])
    model = build_model(config, vocab, num_cls)
    model.load_state_dict(load_safetensor_state_dict(checkpoint_path), strict=False)

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config_path,
    )

    device = torch.device(f"cuda:{model_engine.local_rank}")
    os.makedirs(save_dir, exist_ok=True)
    inf_epoch(model_engine, test_loader, device, save_dir)


if __name__ == "__main__":
    main()



# nohup deepspeed --include localhost:0,1,2,3,4,5 --master_port 29611 infer.py --dataset pancreas16k --exp-id 2026-06-17-17-52 > infer.log 2>&1 &