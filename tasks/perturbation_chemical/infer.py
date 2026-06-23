import argparse
import os

import deepspeed
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader

from engine.checkpoint import load_safetensor_state_dict
from engine.collator import scCollatorPerturbDrugCls
from engine.get_vocab import get_vocab
from engine.model import TransformerModel
from engine.run_epoch import inf_epoch


DATASET_CONFIGS = {
    "tahoe100M_plate1": {
        "test_file": "../../data/tahoe100M-plate1/test.parquet",
        "vocab_path": "../../meta_info/vocab.json",
        "gene_order_path": "../../meta_info/gene_order.pkl",
        "gene_list_path": "../../data/tahoe100M-plate1/gene_name_list.json",
        "max_length": 1601,
        "num_layers": 12,
        "embsize": 512,
        "nhead": 8,
        "dim_feedforward": 2048,
        "moe_experts": 6,
        "epsize": 1,
        "top_k": 2,
        "batch_size": 8,
        "log_interval": 100,
        "nonzero_only_gene": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for the unified perturbation chemical task.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Injected by deepspeed launcher.")
    parser.add_argument("--dataset", default="tahoe100M_plate1", choices=sorted(DATASET_CONFIGS.keys()))
    parser.add_argument("--exp-id", required=True, help="Experiment directory name under save/<dataset>/.")
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def build_model(config, vocab):
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
        mlm=False,
        cls=False,
        pert_drug=True,
        dim_feedforward=config["dim_feedforward"],
        moe_experts=config["moe_experts"],
        epsize=config["epsize"],
        top_k=config["top_k"],
    )


def main():
    args = parse_args()
    config = DATASET_CONFIGS[args.dataset]

    ds_config_path = "../../ckpt_scPoly/ds_config.json"
    save_dir = os.path.join("./save", args.dataset, args.exp_id)
    checkpoint_path = os.path.join(save_dir, "model.safetensors")
    batch_size = args.batch_size or config["batch_size"]

    test_dataset = load_dataset("parquet", data_files=config["test_file"], split="train")
    collator = scCollatorPerturbDrugCls(
        max_length=config["max_length"],
        vocab_path=config["vocab_path"],
        gene_order_path=config["gene_order_path"],
        gene_list_path=config["gene_list_path"],
        mask_exp=False,
        verbose=False,
        inf=True,
        nonzero_only_gene=config["nonzero_only_gene"],
        cls=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    vocab = get_vocab(config["vocab_path"])
    model = build_model(config, vocab)
    model.load_state_dict(load_safetensor_state_dict(checkpoint_path), strict=False)

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config_path,
    )

    device = torch.device(f"cuda:{model_engine.local_rank}")
    os.makedirs(save_dir, exist_ok=True)
    inf_epoch(model_engine, dataloader=test_loader, device=device, log_interval=config["log_interval"], save_dir=save_dir)


if __name__ == "__main__":
    main()



# nohup deepspeed --include localhost:0,1,2,3,4,5 --master_port 29631 infer.py --dataset tahoe100M_plate1 --exp-id 2026-06-22-12-00 > infer.log 2>&1 &
