import json
import pickle

import numpy as np
import torch


class scCollatorPerturb:
    def __init__(
        self,
        max_length,
        pad_id=0,
        cls_id=1,
        pad_exp=-2,
        cls_exp=0,
        cls_chr=0,
        cls_pos=0,
        pad_chr=26,
        pad_pos=-2,
        mask_ratio=0.15,
        mask_exp=False,
        mask_chr=False,
        mask_pos=False,
        mask_value=-1,
        pad_pert=2,
        vocab_path=None,
        gene_order_path=None,
        gene_list_path=None,
        verbose=True,
        inf=False,
        nonzero_only_gene=False,
    ):
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
        self.pad_pert = pad_pert
        self.inf = inf
        self.nonzero_only_gene = nonzero_only_gene

        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        self.id2gene = {v: k for k, v in vocab.items()}

        with open(gene_list_path, "r") as f:
            gene_names = json.load(f)
        self.gene_names = gene_names
        self.gene_ids = [vocab.get(gene, 0) for gene in gene_names]

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

    def _collect_gene_meta(self, gene_ids, missing_gene_ids_all, missing_gene_names_all):
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
        return chr_ids, pos_ids

    def __call__(self, batch):
        missing_gene_ids_all = set()
        missing_gene_names_all = set()
        batch_gene_ids = []
        batch_ctrl_exps = []
        batch_chr_ids = []
        batch_pos_ids = []
        batch_target_exps = []
        batch_pert_mtx = []
        batch_de_mask = []
        batch_de_indices = []
        batch_ctrl_exp_tensor_de = []
        batch_target_exp_tensor_de = []
        batch_masked_exps = []
        batch_mask_bool = []

        for item in batch:
            gene_ids = self.gene_ids
            ctrl_exp = item["ctrl_exp"]
            target_exp = item["target_exp"]
            pert_mtx = item["pert_mtx"]
            pert_idx = item["pert_idx"]
            de_idx = item["de_idx"] if not self.inf else item["de_idx_nondropout"]

            if isinstance(gene_ids, torch.Tensor):
                gene_ids = gene_ids.tolist()
            if isinstance(ctrl_exp, torch.Tensor):
                ctrl_exp = ctrl_exp.tolist()
            if isinstance(target_exp, torch.Tensor):
                target_exp = target_exp.tolist()
            if isinstance(pert_mtx, torch.Tensor):
                pert_mtx = pert_mtx.tolist()
            if isinstance(pert_idx, torch.Tensor):
                pert_idx = pert_idx.tolist()
            if isinstance(de_idx, torch.Tensor):
                de_idx = de_idx.tolist()

            chr_ids, pos_ids = self._collect_gene_meta(gene_ids, missing_gene_ids_all, missing_gene_names_all)
            required_indices = set(de_idx) | set(pert_idx)

            if len(required_indices) > self.max_length:
                raise ValueError(
                    f"required indices ({len(required_indices)}) exceed max_length ({self.max_length})"
                )

            if self.nonzero_only_gene:
                nonzero_indices = {i for i, val in enumerate(ctrl_exp) if val != 0}
                combined_indices = sorted(nonzero_indices | required_indices)

                if len(combined_indices) > self.max_length:
                    rng_2 = np.random.RandomState(None)
                    remaining_indices = list(set(combined_indices) - required_indices)
                    num_to_sample = self.max_length - len(required_indices)
                    sampled = rng_2.choice(remaining_indices, num_to_sample, replace=False)
                    final_indices = sorted(list(required_indices) + list(sampled))
                else:
                    final_indices = combined_indices
            else:
                if len(gene_ids) > self.max_length:
                    rng_2 = np.random.RandomState(None)
                    all_indices = list(range(len(gene_ids)))
                    remaining_indices = list(set(all_indices) - required_indices)
                    num_to_sample = self.max_length - len(required_indices)
                    sampled = rng_2.choice(remaining_indices, num_to_sample, replace=False)
                    final_indices = sorted(list(required_indices) + list(sampled))
                else:
                    final_indices = list(range(len(gene_ids)))

            assert len(gene_ids) == len(ctrl_exp) == len(target_exp) == len(pert_mtx) == len(chr_ids) == len(pos_ids), "长度不一致，数据异常！"

            gene_ids = [gene_ids[i] for i in final_indices]
            ctrl_exp = [ctrl_exp[i] for i in final_indices]
            target_exp = [target_exp[i] for i in final_indices]
            pert_mtx = [pert_mtx[i] for i in final_indices]
            chr_ids = [chr_ids[i] for i in final_indices]
            pos_ids = [pos_ids[i] for i in final_indices]

            pad_len = self.max_length - len(final_indices)
            if pad_len > 0:
                gene_ids += [self.pad_id] * pad_len
                ctrl_exp += [self.pad_exp] * pad_len
                target_exp += [self.pad_exp] * pad_len
                chr_ids += [self.pad_chr] * pad_len
                pert_mtx += [self.pad_pert] * pad_len
                pos_ids += [self.pad_pos] * pad_len

            gene_ids_tensor = torch.tensor(gene_ids, dtype=torch.long)
            ctrl_exp_tensor = torch.tensor(ctrl_exp, dtype=torch.float)
            target_exp_tensor = torch.tensor(target_exp, dtype=torch.float)
            chr_ids_tensor = torch.tensor(chr_ids, dtype=torch.long)
            pos_ids_tensor = torch.tensor(pos_ids, dtype=torch.float)
            pert_mtx_tensor = torch.tensor(pert_mtx, dtype=torch.float)

            idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(final_indices)}
            de_gene_tensor_indices = torch.tensor([idx_map[i] for i in de_idx if i in idx_map], dtype=torch.long)
            de_mask_tensor = torch.zeros(self.max_length, dtype=torch.bool)
            if len(de_gene_tensor_indices) > 0:
                de_mask_tensor[de_gene_tensor_indices] = True
            ctrl_exp_tensor_de = ctrl_exp_tensor[de_gene_tensor_indices]
            target_exp_tensor_de = target_exp_tensor[de_gene_tensor_indices]

            batch_gene_ids.append(gene_ids_tensor)
            batch_ctrl_exps.append(ctrl_exp_tensor)
            batch_chr_ids.append(chr_ids_tensor)
            batch_pos_ids.append(pos_ids_tensor)
            batch_target_exps.append(target_exp_tensor)
            batch_pert_mtx.append(pert_mtx_tensor)
            batch_de_mask.append(de_mask_tensor)
            batch_de_indices.append(de_gene_tensor_indices)
            batch_ctrl_exp_tensor_de.append(ctrl_exp_tensor_de)
            batch_target_exp_tensor_de.append(target_exp_tensor_de)

            if self.mask_exp:
                masked_exps_tensor, exp_mask_tensor = self.random_mask(
                    target_exp_tensor, self.mask_ratio, self.mask_value, self.pad_exp, cls_index=0
                )
                batch_masked_exps.append(masked_exps_tensor)
                batch_mask_bool.append(exp_mask_tensor)

        if self.verbose:
            if missing_gene_ids_all:
                print(f"[Batch Warning] {len(missing_gene_ids_all)} gene_ids not found in vocab: {sorted(missing_gene_ids_all)}")
            if missing_gene_names_all:
                print(f"[Batch Warning] {len(missing_gene_names_all)} gene_names not found in gene_order: {sorted(missing_gene_names_all)}")

        batch_padding_mask = [ids == self.pad_id for ids in batch_gene_ids]

        out = {
            "gene_id": torch.stack(batch_gene_ids),
            "ctrl_exp": torch.stack(batch_ctrl_exps),
            "chr": torch.stack(batch_chr_ids),
            "pos": torch.stack(batch_pos_ids),
            "src_key_padding_mask": torch.stack(batch_padding_mask),
            "target_exp": torch.stack(batch_target_exps),
            "pert_mtx": torch.stack(batch_pert_mtx),
            "de_mask": torch.stack(batch_de_mask),
            "de_idx": torch.stack(batch_de_indices),
            "ctrl_exp_de": torch.stack(batch_ctrl_exp_tensor_de),
            "target_exp_de": torch.stack(batch_target_exp_tensor_de),
        }

        if self.mask_exp:
            out.update(
                {
                    "masked_exp": torch.stack(batch_masked_exps),
                    "exp_mask_bool": torch.stack(batch_mask_bool),
                }
            )

        return out
