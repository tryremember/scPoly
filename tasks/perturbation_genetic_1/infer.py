import argparse
import os

import deepspeed
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader

from engine.checkpoint import load_safetensor_state_dict
from engine.collator import scCollatorPerturb
from engine.get_vocab import get_vocab
from engine.model import TransformerModel
from engine.run_epoch import inf_epoch


DATASET_CONFIGS = {
    "adamson": {
        "vocab_path": "../../data/adamson/vocab_custom.json",
        "gene_list_path": "../../data/adamson/gene_name_list.json",
        "test_file": "../../data/adamson/test.parquet",
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
        "nonzero_only_gene": False,
    },
    "norman": {
        "vocab_path": "../../data/norman/vocab_custom.json",
        "gene_list_path": "../../data/norman/gene_name_list.json",
        "test_file": "../../data/norman/test.parquet",
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
        "nonzero_only_gene": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for the unified perturbation task.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Injected by deepspeed launcher.")
    parser.add_argument("--dataset", default="adamson", choices=sorted(DATASET_CONFIGS.keys()), help="Inference dataset.")
    parser.add_argument(
        "--train-dataset",
        default=None,
        choices=sorted(DATASET_CONFIGS.keys()),
        help="Training dataset that produced the checkpoint. Defaults to --dataset.",
    )
    parser.add_argument("--exp-id", required=True, help="Experiment directory name under save/<train_dataset>/.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-name", default=None, help="Optional subdirectory under the experiment save dir.")
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


def main():
    args = parse_args()
    infer_config = DATASET_CONFIGS[args.dataset]
    train_dataset = args.train_dataset or args.dataset
    train_config = DATASET_CONFIGS[train_dataset]

    ds_config_path = "../../ckpt_scPoly/ds_config.json"
    exp_dir = os.path.join("./save", train_dataset, args.exp_id)
    checkpoint_path = os.path.join(exp_dir, "model.safetensors")
    batch_size = args.batch_size or infer_config["batch_size"]

    if args.output_name is not None:
        save_dir = os.path.join(exp_dir, args.output_name)
    elif train_dataset != args.dataset:
        save_dir = os.path.join(exp_dir, args.dataset)
    else:
        save_dir = exp_dir

    test_dataset = load_dataset("parquet", data_files=infer_config["test_file"], split="train")

    collator = scCollatorPerturb(
        max_length=infer_config["max_length"],
        vocab_path=train_config["vocab_path"],
        gene_order_path=infer_config["gene_order_path"],
        gene_list_path=infer_config["gene_list_path"],
        mask_exp=False,
        verbose=False,
        inf=True,
        nonzero_only_gene=infer_config["nonzero_only_gene"],
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    vocab = get_vocab(train_config["vocab_path"])
    model = build_model(train_config, vocab)
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



# nohup deepspeed --include localhost:0,1,2,3,4,5 --master_port 29622 infer.py --dataset adamson --exp-id 2026-06-22-12-00 > infer.log 2>&1 &
