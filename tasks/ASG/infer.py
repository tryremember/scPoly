import argparse
import os
import deepspeed
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from engine.checkpoint import load_safetensor_state_dict
from engine.collator import scCollatorASG
from engine.data_utils import h5ad2dataset, preprocess_h5ad
from engine.get_vocab import get_vocab
from engine.model import TransformerModel
from engine.run_epoch import inf_epoch


DATASET_CONFIGS = {
    "ms_23en_ctrl": {
        "h5ad_path": "../../data/ms/23en/23en_ctrl.h5ad",
        "vocab_path": "../../data/ms/23en/vocab_custom.json",
        "gene_order_path": "../../meta_info/gene_order.pkl",
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
        "do_norm_and_log1p": False,
        "frozen_gene": True,
    },
    "ms_23en_ms": {
        "h5ad_path": "../../data/ms/23en/23en_ms.h5ad",
        "vocab_path": "../../data/ms/23en/vocab_custom.json",
        "gene_order_path": "../../meta_info/gene_order.pkl",
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
        "do_norm_and_log1p": False,
        "frozen_gene": True,
    },
    "ms_opc_ctrl": {
        "h5ad_path": "../../data/ms/opc/opc_ctrl.h5ad",
        "vocab_path": "../../data/ms/opc/vocab_custom.json",
        "gene_order_path": "../../meta_info/gene_order.pkl",
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
        "do_norm_and_log1p": False,
        "frozen_gene": True,
    },
    "ms_opc_ms": {
        "h5ad_path": "../../data/ms/opc/opc_ms.h5ad",
        "vocab_path": "../../data/ms/opc/vocab_custom.json",
        "gene_order_path": "../../meta_info/gene_order.pkl",
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
        "do_norm_and_log1p": False,
        "frozen_gene": True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for the unified ASG task.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Injected by deepspeed launcher.")
    parser.add_argument("--dataset", default="ms_23en_ctrl", choices=sorted(DATASET_CONFIGS.keys()))
    parser.add_argument("--exp-id", required=True, help="Experiment directory name under save/<dataset>/.")
    parser.add_argument("--tag", default="default", help="Suffix used for the saved attention-weight filename.")
    parser.add_argument("--h5ad-path", default=None, help="Optional override for the inference h5ad file.")
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
        dim_feedforward=config["dim_feedforward"],
        moe_experts=config["moe_experts"],
        epsize=config["epsize"],
        top_k=config["top_k"],
        moe_layers=config["moe_layers"],
    )


def main():
    args = parse_args()
    config = DATASET_CONFIGS[args.dataset]

    ds_config_path = "../../ckpt_scPoly/ds_config.json"
    save_dir = os.path.join("./save", args.dataset, args.exp_id)
    checkpoint_path = os.path.join(save_dir, "model.safetensors")
    batch_size = args.batch_size or config["batch_size"]
    h5ad_path = args.h5ad_path or config["h5ad_path"]

    adata = preprocess_h5ad(h5ad_path, do_norm_and_log1p=config["do_norm_and_log1p"])
    dataset, _, _ = h5ad2dataset(adata, config["vocab_path"], frozen_gene=config["frozen_gene"], inf=True)

    collator = scCollatorASG(
        max_length=config["max_length"],
        vocab_path=config["vocab_path"],
        gene_order_path=config["gene_order_path"],
        mask_exp=False,
        verbose=False,
        inf=True,
        frozen_gene=config["frozen_gene"],
    )
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

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
    inf_epoch(model_engine, test_loader, device, save_dir, args.tag)


if __name__ == "__main__":
    main()



# nohup deepspeed --include localhost:0,1,2,3,4,5 --master_port 29641 infer.py --dataset ms_23en_ctrl --exp-id 2026-06-22-12-00 --tag 23en_ctrl > infer.log 2>&1 &
