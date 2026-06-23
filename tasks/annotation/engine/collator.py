import json
import pickle
import numpy as np
import pickle
import torch
from .model import *
from .get_vocab import *
import numpy as np
import json


class scCollator:
    """
    Collate tokenized single-cell records into fixed-length tensors for model input.

    Supports two modes via `inf`:
    - inf=False: expects labels (celltype_id, batch_id) and includes them in the batch.
    - inf=True: inference mode; skips label handling and returns expression-only batch.

    Each sample is expected to provide:
    - gene_id: sequence of gene token IDs
    - exp: sequence of expression values aligned with gene_id
    - cell_id: integer cell identifier
    - (optional, when inf=False) celltype_id, batch_id

    The collator:
    - maps gene_id -> gene_name -> (chr, pos) using external metadata
    - prepends a <cls> token across all feature streams
    - truncates (random subset) or pads sequences to `max_length`
    - optionally masks expression values for masked prediction objectives
    """
    def __init__(self, max_length, pad_id=0, cls_id=1, pad_exp=-2, cls_exp=0, cls_chr=0, cls_pos=0, pad_chr = 26, pad_pos = -2, 
                 mask_ratio=0.15, mask_exp=False, mask_chr=False, mask_pos=False, mask_value=-1,
                 vocab_path=None, gene_order_path=None, verbose=True, inf=False):
        self.max_length = max_length
        self.pad_id = pad_id
        self.cls_id = cls_id
        self.pad_exp = pad_exp
        self.cls_exp = cls_exp
        self.cls_chr = cls_chr
        self.cls_pos = cls_pos
        self.pad_chr = pad_chr
        self.pad_pos = pad_pos
        self.verbose = verbose
        self.mask_ratio = mask_ratio
        self.mask_exp = mask_exp
        self.mask_chr = mask_chr
        self.mask_pos = mask_pos
        self.mask_value = mask_value
        self.inf = inf


        # gene_id -> gene_name
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        self.id2gene = {v: k for k, v in vocab.items()}


        # gene_name -> chr, pos
        with open(gene_order_path, "rb") as f:
            gene_info_list = pickle.load(f)
        self.gene_meta = {g["gene_name"]: {"chr": int(g["chr"]), "pos": int(g["pos"])} for g in gene_info_list}

    def random_mask(self, values, mask_ratio, mask_value, pad_value, cls_index, seed=None):
        """
        Randomly mask values in a 1D sequence, excluding pad positions and the CLS index.

        Parameters
        ----------
        values : torch.Tensor or np.ndarray
            1D sequence to mask.
        mask_ratio : float
            Fraction of eligible positions to mask.
        mask_value : numeric
            Value used to replace masked positions.
        pad_value : numeric
            Pad sentinel; pad positions are never masked.
        cls_index : int
            Index position of the CLS token in the sequence (excluded from masking).
        seed : int, optional
            Seed for deterministic masking.

        Returns
        -------
        masked_values : torch.Tensor
            Masked sequence (float tensor).
        mask_flags : torch.Tensor
            Boolean mask indicating which positions were masked.
        """
        values = values.reshape(1, -1)
        if isinstance(values, torch.Tensor):
            values = values.clone().detach().numpy()
        else:
            values = values.copy()

        mask_flags = np.zeros_like(values, dtype=np.bool_)

        rng = np.random.RandomState(seed)

        for i in range(len(values)):
            row = values[i]
            idx_can_be_masked = np.nonzero(row != pad_value)[0]
            idx_can_be_masked = idx_can_be_masked[idx_can_be_masked != cls_index]
            n_mask = int(len(idx_can_be_masked) * mask_ratio)
            if n_mask > 0:
                mask_idx = rng.choice(idx_can_be_masked, n_mask, replace=False)
                row[mask_idx] = mask_value
                mask_flags[i, mask_idx] = True
        return torch.from_numpy(values).float(), torch.from_numpy(mask_flags)

    def __call__(self, batch):
        """
        Collate a list of per-cell token records into a model-ready batch dict.

        Returns
        -------
        dict
            Always includes:
            - cell_id, gene_id, exp, chr, pos, src_key_padding_mask, target_exp
            Optionally includes (when inf=False):
            - celltype_id, batch_id
            Optionally includes (when mask_exp=True):
            - masked_exp, exp_mask_bool
        """
        missing_gene_ids_all = set()
        missing_gene_names_all = set()

        batch_gene_ids, batch_exps = [], []
        batch_chr_ids, batch_pos_ids = [], []
        batch_target_exps, batch_masked_exps, batch_mask_bool = [], [], []

        cell_ids = []
        celltype_ids = []
        batch_ids = []

        for item in batch:
            gene_ids = item["gene_id"]
            exps = item["exp"]

            if isinstance(gene_ids, torch.Tensor):
                gene_ids = gene_ids.tolist()
            if isinstance(exps, torch.Tensor):
                exps = exps.tolist()

            # gene_id -> gene_name -> (chr, pos)
            chr_ids = []
            pos_ids = []
            for gid in gene_ids:
                gene_name = self.id2gene.get(gid, "<unk>")
                if gene_name == "<unk>":
                    missing_gene_ids_all.add(gid)
                meta = self.gene_meta.get(gene_name, {"chr": self.pad_chr, "pos": self.pad_pos})
                if meta["chr"] == self.pad_chr and meta["pos"] == self.pad_pos:
                    missing_gene_names_all.add(gene_name)
                chr_ids.append(meta["chr"])
                pos_ids.append(meta["pos"])

            # prepend CLS
            gene_ids = [self.cls_id] + gene_ids
            exps = [self.cls_exp] + exps
            chr_ids = [self.cls_chr] + chr_ids
            pos_ids = [self.cls_pos] + pos_ids

            # truncate or pad
            if len(gene_ids) > self.max_length:
                rng_2 = np.random.RandomState(None)
                indices = list(range(1, len(gene_ids)))
                selected = rng_2.choice(indices, self.max_length - 1, replace=False)
                selected.sort()

                assert len(gene_ids) == len(exps) == len(chr_ids) == len(pos_ids), "Length mismatch."

                gene_ids = [gene_ids[0]] + [gene_ids[i] for i in selected]
                exps = [exps[0]] + [exps[i] for i in selected]
                chr_ids = [chr_ids[0]] + [chr_ids[i] for i in selected]
                pos_ids = [pos_ids[0]] + [pos_ids[i] for i in selected]
            else:
                pad_len = self.max_length - len(gene_ids)
                gene_ids += [self.pad_id] * pad_len
                exps += [self.pad_exp] * pad_len
                chr_ids += [self.pad_chr] * pad_len
                pos_ids += [self.pad_pos] * pad_len

            gene_ids_tensor = torch.tensor(gene_ids, dtype=torch.long)
            exps_tensor = torch.tensor(exps, dtype=torch.float)
            chr_ids_tensor = torch.tensor(chr_ids, dtype=torch.long)
            pos_ids_tensor = torch.tensor(pos_ids, dtype=torch.float)

            target_exp_tensor = exps_tensor.clone()

            if self.mask_exp:
                masked_exps_tensor, exp_mask_tensor = self.random_mask(
                    exps_tensor, self.mask_ratio, self.mask_value, self.pad_exp, cls_index=0
                )

            batch_gene_ids.append(gene_ids_tensor)
            batch_exps.append(exps_tensor)
            batch_chr_ids.append(chr_ids_tensor)
            batch_pos_ids.append(pos_ids_tensor)
            batch_target_exps.append(target_exp_tensor)

            if self.mask_exp:
                batch_masked_exps.append(masked_exps_tensor)
                batch_mask_bool.append(exp_mask_tensor)

            cell_ids.append(item["cell_id"])

            if not self.inf:
                celltype_ids.append(item["celltype_id"])
                batch_ids.append(item["batch_id"])

        if self.verbose:
            if missing_gene_ids_all:
                print(f"[Batch Warning] {len(missing_gene_ids_all)} gene_ids not found in vocab: {sorted(missing_gene_ids_all)}")
            if missing_gene_names_all:
                print(f"[Batch Warning] {len(missing_gene_names_all)} gene_names not found in gene_order: {sorted(missing_gene_names_all)}")

        batch_padding_mask = [ids == self.pad_id for ids in batch_gene_ids]

        out = {
            "cell_id": torch.tensor(cell_ids, dtype=torch.long),
            "gene_id": torch.stack(batch_gene_ids),
            "exp": torch.stack(batch_exps),
            "chr": torch.stack(batch_chr_ids),
            "pos": torch.stack(batch_pos_ids),
            "src_key_padding_mask": torch.stack(batch_padding_mask),
            "target_exp": torch.stack(batch_target_exps),
        }

        if not self.inf:
            out.update({
                "celltype_id": torch.tensor(celltype_ids, dtype=torch.long),
                "batch_id": torch.tensor(batch_ids, dtype=torch.long),
            })

        if self.mask_exp:
            out.update({
                "masked_exp": torch.stack(batch_masked_exps),
                "exp_mask_bool": torch.stack(batch_mask_bool),
            })

        return out
