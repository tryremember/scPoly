import json

import numpy as np
import scanpy as sc
import torch
from datasets import Dataset
from scipy.sparse import csr_matrix, issparse, vstack


def do_normalizing_and_log1p(adata):
    print("Normalizing ...")
    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    print("log1p ...")
    sc.pp.log1p(adata)
    return adata


def do_binning_fast(adata, n_bins):
    print("binning data ...")
    layer_data = adata.X
    n_rows = layer_data.shape[0]
    binned_sparse_rows = []
    bin_edges_sparse_rows = []

    for i in range(n_rows):
        row = layer_data[i].toarray().ravel() if issparse(layer_data) else layer_data[i]

        if row.min() < 0:
            raise ValueError(f"Assuming non-negative data, but got min value {row.min()} in row {i}.")

        if row.max() == 0:
            binned_sparse_rows.append(csr_matrix(np.zeros_like(row, dtype=np.int64)))
            bin_edges_sparse_rows.append(csr_matrix([0] * n_bins))
            continue

        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        non_zero_digits = np.digitize(non_zero_row, bins)

        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
        binned_sparse_rows.append(csr_matrix(binned_row))
        bin_edges_sparse_rows.append(csr_matrix(np.concatenate([[0], bins])))

        if i % 10000 == 0:
            print(f"[debug] processed {i + 1}/{n_rows} rows")

    adata.layers["X_binned"] = vstack(binned_sparse_rows)
    adata.obsm["bin_edges"] = vstack(bin_edges_sparse_rows)
    print("[debug] binning done")
    return adata


def preprocess_h5ad(h5ad_path, do_norm_and_log1p=False):
    adata = sc.read_h5ad(h5ad_path)
    if do_norm_and_log1p:
        adata = do_normalizing_and_log1p(adata)
    adata = do_binning_fast(adata, 51)
    return adata


def h5ad2dataset(adata, vocab_path, frozen_gene=True, inf=False):
    adata.var["gene_name"] = adata.var.index.tolist()

    with open(vocab_path, "r") as f:
        name2int = json.load(f)

    pe_gene_set = set(name2int.keys())
    genes = adata.var["gene_name"]
    matched_mask = genes.isin(pe_gene_set)
    matched_genes = genes[matched_mask]
    adata = adata[:, matched_mask].copy()

    matched_count = matched_genes.shape[0]
    vocab_size = len(pe_gene_set)
    print(f"Matched {matched_count} genes out of {vocab_size} in vocab ({matched_count / vocab_size:.2%})")

    genes = adata.var["gene_name"]
    gene2id = {name: name2int.get(name, -404) for name in genes}
    gene_names_array = genes.to_numpy()

    if not inf:
        adata.obs["celltype"] = adata.obs["celltype"].astype("category")
        adata.obs["batch"] = adata.obs["batch"].astype("category")
        adata.obs["celltype_code"] = adata.obs["celltype"].cat.codes
        adata.obs["batch_code"] = adata.obs["batch"].cat.codes
        num_celltype = len(adata.obs["celltype"].cat.categories)
        num_batch = len(adata.obs["batch"].cat.categories)
        celltype2int = {cat: i for i, cat in enumerate(adata.obs["celltype"].cat.categories)}
        batch2int = {cat: i for i, cat in enumerate(adata.obs["batch"].cat.categories)}
    else:
        num_celltype = None
        num_batch = None
        celltype2int = None
        batch2int = None

    tokenized_data = []
    for i in range(adata.layers["X_binned"].shape[0]):
        row = adata.layers["X_binned"][i]
        if frozen_gene:
            row_dense = row.toarray().flatten() if hasattr(row, "toarray") else row.flatten()
            gene_ids = [gene2id[name] for name in gene_names_array]
            expressions = row_dense
        else:
            idx = row.indices
            expressions = row.data
            gene_ids = [gene2id[name] for name in gene_names_array[idx]]

        item = {
            "cell_id": i,
            "gene_id": gene_ids,
            "exp": expressions.tolist() if isinstance(expressions, np.ndarray) else expressions,
        }

        if not inf:
            item["celltype_id"] = celltype2int[adata.obs["celltype"].iloc[i]]
            item["batch_id"] = batch2int[adata.obs["batch"].iloc[i]]

        tokenized_data.append(item)

    return Dataset.from_list(tokenized_data), num_celltype, num_batch
