import json
import pickle

import numpy as np
import torch


class scCollatorPertGenetic:
    def __init__(
        self,
        max_length,
        pad_id=0,
        cls_id=1,
        pad_exp=-2,
        cls_exp=0,
        cls_chr=0,
        cls_pos=0,
        cls_pert=0,
        pad_chr=26,
        pad_pos=-2,
        mask_ratio=0.05,
        mask_ratio_2=0.3,
        mask_exp=True,
        mask_chr=False,
        mask_pos=False,
        mask_value=-1,
        frozen_gene=True,
        vocab_path=None,
        gene_order_path=None,
        verbose=False,
        inf=False,
        CL=False,
    ):
        self.max_length = max_length
        self.pad_id = pad_id
        self.cls_id = cls_id
        self.pad_exp = pad_exp
        self.cls_exp = cls_exp
        self.cls_chr = cls_chr
        self.cls_pos = cls_pos
        self.cls_pert = cls_pert
        self.pad_chr = pad_chr
        self.pad_pos = pad_pos
        self.verbose = verbose
        self.mask_ratio_1 = mask_ratio
        self.mask_ratio_2 = mask_ratio_2
        self.mask_exp = mask_exp
        self.mask_chr = mask_chr
        self.mask_pos = mask_pos
        self.mask_value = mask_value
        self.inf = inf
        self.frozen_gene = frozen_gene
        self.contrastive_learning = CL

        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        self.id2gene = {v: k for k, v in vocab.items()}

        with open(gene_order_path, "rb") as f:
            gene_info_list = pickle.load(f)
        self.gene_meta = {g["gene_name"]: {"chr": int(g["chr"]), "pos": int(g["pos"])} for g in gene_info_list}

    def random_mask(self, values, mask_ratio, mask_value, pad_value, cls_index, seed=None):
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
        missing_gene_ids_all = set()
        missing_gene_names_all = set()
        batch_gene_ids = []
        batch_chr_ids = []
        batch_pos_ids = []
        batch_target_exps = []
        batch_masked_exps_1 = []
        batch_mask_bool_1 = []
        batch_masked_exps_2 = []
        batch_mask_bool_2 = []
        batch_ctrl_exps = []
        batch_pert_mtx = []

        for item in batch:
            gene_ids = item["gene_id"]
            ctrl_exp = item["ctrl_exp"]
            target_exp = item["target_exp"]
            pert_mtx = [0] * 1700
            pert_mtx[-1] = 1

            if isinstance(gene_ids, torch.Tensor):
                gene_ids = gene_ids.tolist()
            if isinstance(ctrl_exp, torch.Tensor):
                ctrl_exp = ctrl_exp.tolist()
            if isinstance(target_exp, torch.Tensor):
                target_exp = target_exp.tolist()

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

            gene_ids = [self.cls_id] + gene_ids
            chr_ids = [self.cls_chr] + chr_ids
            pos_ids = [self.cls_pos] + pos_ids
            ctrl_exp = [self.cls_exp] + ctrl_exp
            target_exp = [self.cls_exp] + target_exp
            pert_mtx = [self.cls_pert] + pert_mtx

            if self.frozen_gene:
                assert len(gene_ids) == len(ctrl_exp) == len(chr_ids) == len(pos_ids) == len(target_exp) == len(pert_mtx), "Inconsistent sequence lengths."
            else:
                raise NotImplementedError("Only frozen_gene=True is kept in this organized main task version.")

            gene_ids_tensor = torch.tensor(gene_ids, dtype=torch.long)
            ctrl_exp_tensor = torch.tensor(ctrl_exp, dtype=torch.float)
            target_exp_tensor = torch.tensor(target_exp, dtype=torch.float)
            chr_ids_tensor = torch.tensor(chr_ids, dtype=torch.long)
            pos_ids_tensor = torch.tensor(pos_ids, dtype=torch.float)
            pert_mtx_tensor = torch.tensor(pert_mtx, dtype=torch.float)

            if self.mask_exp:
                masked_exps_tensor_1, exp_mask_tensor_1 = self.random_mask(
                    ctrl_exp_tensor, self.mask_ratio_1, self.mask_value, self.pad_exp, cls_index=0
                )
            if self.contrastive_learning:
                masked_exps_tensor_2, exp_mask_tensor_2 = self.random_mask(
                    ctrl_exp_tensor, self.mask_ratio_2, self.mask_value, self.pad_exp, cls_index=0
                )

            batch_gene_ids.append(gene_ids_tensor)
            batch_ctrl_exps.append(ctrl_exp_tensor)
            batch_chr_ids.append(chr_ids_tensor)
            batch_pos_ids.append(pos_ids_tensor)
            batch_target_exps.append(target_exp_tensor)
            batch_pert_mtx.append(pert_mtx_tensor)

            if self.mask_exp:
                batch_masked_exps_1.append(masked_exps_tensor_1)
                batch_mask_bool_1.append(exp_mask_tensor_1)
            if self.contrastive_learning:
                batch_masked_exps_2.append(masked_exps_tensor_2)
                batch_mask_bool_2.append(exp_mask_tensor_2)

        if self.verbose:
            if missing_gene_ids_all:
                print(f"[Batch Warning] {len(missing_gene_ids_all)} gene_ids not found in vocab: {sorted(missing_gene_ids_all)}")
            if missing_gene_names_all:
                print(f"[Batch Warning] {len(missing_gene_names_all)} gene_names not found in gene_order: {sorted(missing_gene_names_all)}")

        batch_padding_mask = [ids == self.pad_id for ids in batch_gene_ids]

        out = {
            "gene_id": torch.stack(batch_gene_ids),
            "ctrl_exp": torch.stack(batch_ctrl_exps),
            "target_exp": torch.stack(batch_target_exps),
            "pert_mtx": torch.stack(batch_pert_mtx),
            "chr": torch.stack(batch_chr_ids),
            "pos": torch.stack(batch_pos_ids),
            "src_key_padding_mask": torch.stack(batch_padding_mask),
        }
        if self.mask_exp:
            out.update(
                {
                    "masked_exp_1": torch.stack(batch_masked_exps_1),
                    "exp_mask_bool_1": torch.stack(batch_mask_bool_1),
                }
            )
        if self.contrastive_learning:
            out.update(
                {
                    "masked_exp_2": torch.stack(batch_masked_exps_2),
                    "exp_mask_bool_2": torch.stack(batch_mask_bool_2),
                }
            )

        return out
