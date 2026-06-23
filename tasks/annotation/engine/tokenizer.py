from datasets import Dataset
import json
import numpy as np
import torch
from .model import *
from .get_vocab import *



def h5ad2parquet(
    adata,
    vocab_path,
    parquet_save_path=None,
    list_of_dict=True,
    inf=False,
    alpha_method="inverse_freq",
    celltype_key="celltype",
    batch_key="batch",
):
    """
    Convert an AnnData object into tokenized gene-expression sequences and
    optionally serialize them to a Parquet file.

    The function operates in two modes controlled by `inf`.

    Training mode (inf=False):
    - Uses `celltype_key` and `batch_key` from `adata.obs` to generate integer
      labels for supervision.
    - Optionally computes class weights (`alpha`) based on cell type frequency.
    - Returns tokenized expression data together with label mappings and metadata.

    Inference mode (inf=True):
    - Ignores cell type and batch information.
    - Returns only tokenized gene-expression sequences.

    Tokenization details:
    - Genes are filtered to those present in the vocabulary JSON
      (gene_name -> gene_id).
    - Each cell is represented as a sparse sequence constructed from
      `adata.layers["X_binned"]`, collecting non-zero entries as
      (gene_id list, expression value list).

    Parameters
    ----------
    adata : AnnData
        Input AnnData object. Must contain binned expression values in
        `adata.layers["X_binned"]`.
    vocab_path : str
        Path to a JSON file mapping gene names to integer gene IDs.
    parquet_save_path : str, optional
        If provided, the tokenized dataset is written to this Parquet file.
    list_of_dict : bool, optional
        If True, returns a list of per-cell dictionaries (row-wise format).
        If False, returns a dictionary of lists (column-wise format).
    inf : bool, optional
        If True, runs in inference mode and skips label handling.
        If False, runs in training mode and expects label columns in `adata.obs`.
    alpha_method : str or None, optional
        Strategy for computing class weights.
        Currently supported:
        - "inverse_freq": inverse frequency over cell type categories.
        - None: do not compute class weights.
    celltype_key : str, optional
        Column name in `adata.obs` used as the cell type label in training mode.
    batch_key : str, optional
        Column name in `adata.obs` used as the batch label in training mode.

    Returns
    -------
    If inf is True:
        records : list[dict] or dict[str, list]
            Tokenized gene-expression records containing:
            - "cell_id": int
            - "gene_id": list[int]
            - "exp": list[int or float]

    If inf is False:
        tuple
            (
                records,
                num_celltype,
                num_batch,
                celltype2int,
                int2celltype,
                batch2int,
                int2batch,
                alpha,
            )

        where:
        - records : list[dict] or dict[str, list]
            Tokenized records including "celltype_id" and "batch_id".
        - num_celltype : int
            Number of unique cell types.
        - num_batch : int
            Number of unique batches.
        - celltype2int / int2celltype : dict
            Mappings between cell type labels and integer IDs.
        - batch2int / int2batch : dict
            Mappings between batch labels and integer IDs.
        - alpha : torch.Tensor or None
            Optional class-weight tensor aligned with `celltype2int`.
    """

    adata.var["gene_name"] = adata.var.index.tolist()

    # vocab: gene_name -> gene_id
    with open(vocab_path, "r") as f:
        name2int = json.load(f)

    pe_gene_set = set(name2int.keys())
    genes = adata.var["gene_name"]
    matched_mask = genes.isin(pe_gene_set)
    adata = adata[:, matched_mask].copy()

    matched_count = matched_mask.sum()
    vocab_size = len(pe_gene_set)
    print(f"Matched {matched_count} genes out of {vocab_size} in vocab ({matched_count / vocab_size:.2%})")

    genes = adata.var["gene_name"]
    gene2id = {name: name2int.get(name, -404) for name in genes}

    alpha = None
    if not inf:
        adata.obs[celltype_key] = adata.obs[celltype_key].astype("category")
        adata.obs[batch_key] = adata.obs[batch_key].astype("category")

        celltype2int = {cat: i for i, cat in enumerate(adata.obs[celltype_key].cat.categories)}
        int2celltype = {i: cat for cat, i in celltype2int.items()}
        batch2int = {cat: i for i, cat in enumerate(adata.obs[batch_key].cat.categories)}
        int2batch = {i: cat for cat, i in batch2int.items()}

        num_celltype = len(celltype2int)
        num_batch = len(batch2int)

        if alpha_method == "inverse_freq":
            counts = np.array([np.sum(adata.obs[celltype_key] == ct) for ct in adata.obs[celltype_key].cat.categories])
            alpha = 1.0 / (counts + 1e-6)
            alpha = alpha / alpha.sum()
            alpha = torch.tensor(alpha, dtype=torch.float32)

    row_ids, col_inds, values = [], [], []
    celltype_ids, batch_ids = [], []

    gene_names_array = genes.to_numpy()

    for i in range(adata.layers["X_binned"].shape[0]):
        row = adata.layers["X_binned"][i]
        idx = row.indices
        expressions = row.data

        genes_id = [gene2id[name] for name in gene_names_array[idx]]

        row_ids.append(i)
        col_inds.append(genes_id)
        values.append(expressions)

        if not inf:
            celltype_ids.append(celltype2int[adata.obs[celltype_key].iloc[i]])
            batch_ids.append(batch2int[adata.obs[batch_key].iloc[i]])

    tokenized_data = {
        "cell_id": row_ids,
        "gene_id": col_inds,
        "exp": values,
    }

    if not inf:
        tokenized_data["celltype_id"] = celltype_ids
        tokenized_data["batch_id"] = batch_ids

    if parquet_save_path is not None:
        dataset = Dataset.from_dict(tokenized_data)
        df = dataset.to_pandas()
        df.to_parquet(parquet_save_path)
        print(f"saved to parquet file: {parquet_save_path}")

    if list_of_dict:
        records = [
            {k: (v[i].tolist() if isinstance(v[i], np.ndarray) else v[i]) for k, v in tokenized_data.items()}
            for i in range(len(row_ids))
        ]
    else:
        records = tokenized_data

    if inf:
        return records
    else:
        return (
            records,
            num_celltype,
            num_batch,
            celltype2int,
            int2celltype,
            batch2int,
            int2batch,
            alpha,
        )
