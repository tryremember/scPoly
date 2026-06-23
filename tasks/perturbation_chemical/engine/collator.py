import json
import pickle

import numpy as np
import torch


class scCollatorPerturbDrugCls:
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
        cls=None,
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
        self.cls = cls
        self.nonzero_only_gene = nonzero_only_gene

        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        self.id2gene = {v: k for k, v in vocab.items()}

        with open(gene_list_path, "r") as f:
            gene_names = json.load(f)
        self.gene_names = gene_names
        self.gene_ids = [vocab[gene] for gene in gene_names if gene in vocab]

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
        batch_chr_ids = []
        batch_pos_ids = []
        batch_target_exps = []
        batch_ctrl_exps = []
        batch_smiles = []
        batch_sample = []
        batch_masked_exps = []
        batch_mask_bool = []

        for item in batch:
            gene_ids = self.gene_ids
            ctrl_exp = item["ctrl_exp"]
            target_exp = item["target_exp"]
            sample = item["sample"]
            smiles = item["smiles"]
            if isinstance(gene_ids, torch.Tensor):
                gene_ids = gene_ids.tolist()
            if isinstance(ctrl_exp, torch.Tensor):
                ctrl_exp = ctrl_exp.tolist()
            if isinstance(target_exp, torch.Tensor):
                target_exp = target_exp.tolist()

            chr_ids, pos_ids = self._collect_gene_meta(gene_ids, missing_gene_ids_all, missing_gene_names_all)

            if self.cls:
                gene_ids = [self.cls_id] + gene_ids
                ctrl_exp = [self.cls_exp] + ctrl_exp
                target_exp = [self.cls_exp] + target_exp
                chr_ids = [self.cls_chr] + chr_ids
                pos_ids = [self.cls_pos] + pos_ids

            if self.cls:
                if not self.nonzero_only_gene:
                    if len(gene_ids) >= self.max_length:
                        rng_2 = np.random.RandomState(None)
                        all_indices = list(range(1, len(gene_ids)))
                        sampled = rng_2.choice(all_indices, self.max_length - 1, replace=False)
                        final_indices = sorted(sampled.tolist())
                        assert len(gene_ids) == len(ctrl_exp) == len(target_exp) == len(chr_ids) == len(pos_ids)
                        gene_ids = [gene_ids[0]] + [gene_ids[i] for i in final_indices]
                        ctrl_exp = [ctrl_exp[0]] + [ctrl_exp[i] for i in final_indices]
                        target_exp = [target_exp[0]] + [target_exp[i] for i in final_indices]
                        chr_ids = [chr_ids[0]] + [chr_ids[i] for i in final_indices]
                        pos_ids = [pos_ids[0]] + [pos_ids[i] for i in final_indices]
                    else:
                        pad_len = self.max_length - len(gene_ids)
                        gene_ids += [self.pad_id] * pad_len
                        ctrl_exp += [self.pad_exp] * pad_len
                        target_exp += [self.pad_exp] * pad_len
                        chr_ids += [self.pad_chr] * pad_len
                        pos_ids += [self.pad_pos] * pad_len

                else:
                    gene_ids_wo_cls = gene_ids[1:]
                    ctrl_exp_wo_cls = ctrl_exp[1:]
                    target_exp_wo_cls = target_exp[1:]
                    chr_ids_wo_cls = chr_ids[1:]
                    pos_ids_wo_cls = pos_ids[1:]
                    nonzero_indices = [i for i, val in enumerate(ctrl_exp_wo_cls) if val != 0]

                    if len(nonzero_indices) > self.max_length - 1:
                        rng_2 = np.random.RandomState(None)
                        sampled = rng_2.choice(nonzero_indices, self.max_length - 1, replace=False)
                        final_indices = sorted(sampled.tolist())
                        gene_ids = [gene_ids[0]] + [gene_ids_wo_cls[i] for i in final_indices]
                        ctrl_exp = [ctrl_exp[0]] + [ctrl_exp_wo_cls[i] for i in final_indices]
                        target_exp = [target_exp[0]] + [target_exp_wo_cls[i] for i in final_indices]
                        chr_ids = [chr_ids[0]] + [chr_ids_wo_cls[i] for i in final_indices]
                        pos_ids = [pos_ids[0]] + [pos_ids_wo_cls[i] for i in final_indices]
                    else:
                        gene_ids = [gene_ids[0]] + [gene_ids_wo_cls[i] for i in nonzero_indices]
                        ctrl_exp = [ctrl_exp[0]] + [ctrl_exp_wo_cls[i] for i in nonzero_indices]
                        target_exp = [target_exp[0]] + [target_exp_wo_cls[i] for i in nonzero_indices]
                        chr_ids = [chr_ids[0]] + [chr_ids_wo_cls[i] for i in nonzero_indices]
                        pos_ids = [pos_ids[0]] + [pos_ids_wo_cls[i] for i in nonzero_indices]
                        pad_len = self.max_length - len(nonzero_indices) - 1
                        gene_ids += [self.pad_id] * pad_len
                        ctrl_exp += [self.pad_exp] * pad_len
                        target_exp += [self.pad_exp] * pad_len
                        chr_ids += [self.pad_chr] * pad_len
                        pos_ids += [self.pad_pos] * pad_len

            else:
                if not self.nonzero_only_gene:
                    if len(gene_ids) > self.max_length:
                        rng_2 = np.random.RandomState(None)
                        sampled = rng_2.choice(list(range(len(gene_ids))), self.max_length, replace=False)
                        final_indices = sorted(sampled.tolist())
                        assert len(gene_ids) == len(ctrl_exp) == len(target_exp) == len(chr_ids) == len(pos_ids)
                        gene_ids = [gene_ids[i] for i in final_indices]
                        ctrl_exp = [ctrl_exp[i] for i in final_indices]
                        target_exp = [target_exp[i] for i in final_indices]
                        chr_ids = [chr_ids[i] for i in final_indices]
                        pos_ids = [pos_ids[i] for i in final_indices]
                    else:
                        pad_len = self.max_length - len(gene_ids)
                        gene_ids += [self.pad_id] * pad_len
                        ctrl_exp += [self.pad_exp] * pad_len
                        target_exp += [self.pad_exp] * pad_len
                        chr_ids += [self.pad_chr] * pad_len
                        pos_ids += [self.pad_pos] * pad_len
                else:
                    nonzero_indices = [i for i, val in enumerate(ctrl_exp) if val != 0]
                    if len(nonzero_indices) > self.max_length:
                        rng_2 = np.random.RandomState(None)
                        sampled = rng_2.choice(nonzero_indices, self.max_length, replace=False)
                        final_indices = sorted(sampled.tolist())
                        gene_ids = [gene_ids[i] for i in final_indices]
                        ctrl_exp = [ctrl_exp[i] for i in final_indices]
                        target_exp = [target_exp[i] for i in final_indices]
                        chr_ids = [chr_ids[i] for i in final_indices]
                        pos_ids = [pos_ids[i] for i in final_indices]
                    else:
                        gene_ids = [gene_ids[i] for i in nonzero_indices]
                        ctrl_exp = [ctrl_exp[i] for i in nonzero_indices]
                        target_exp = [target_exp[i] for i in nonzero_indices]
                        chr_ids = [chr_ids[i] for i in nonzero_indices]
                        pos_ids = [pos_ids[i] for i in nonzero_indices]
                        pad_len = self.max_length - len(nonzero_indices)
                        gene_ids += [self.pad_id] * pad_len
                        ctrl_exp += [self.pad_exp] * pad_len
                        target_exp += [self.pad_exp] * pad_len
                        chr_ids += [self.pad_chr] * pad_len
                        pos_ids += [self.pad_pos] * pad_len

            gene_ids_tensor = torch.tensor(gene_ids, dtype=torch.long)
            ctrl_exp_tensor = torch.tensor(ctrl_exp, dtype=torch.float)
            target_exp_tensor = torch.tensor(target_exp, dtype=torch.float)
            chr_ids_tensor = torch.tensor(chr_ids, dtype=torch.long)
            pos_ids_tensor = torch.tensor(pos_ids, dtype=torch.float)
            batch_gene_ids.append(gene_ids_tensor)
            batch_ctrl_exps.append(ctrl_exp_tensor)
            batch_chr_ids.append(chr_ids_tensor)
            batch_pos_ids.append(pos_ids_tensor)
            batch_target_exps.append(target_exp_tensor)
            batch_smiles.append(smiles)
            batch_sample.append(sample)

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
            "smiles": batch_smiles,
            "sample": batch_sample,
        }

        if self.mask_exp:
            out.update(
                {
                    "masked_exp": torch.stack(batch_masked_exps),
                    "exp_mask_bool": torch.stack(batch_mask_bool),
                }
            )

        return out
