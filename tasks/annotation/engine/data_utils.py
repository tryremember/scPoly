from typing import Union
import numpy as np
import torch
from scipy.sparse import issparse,csr_matrix, vstack
import scanpy as sc
from scanpy.get import _get_obs_rep
import pickle


def do_normalizing_and_log1p(adata):
    """
    Normalize and log-transform AnnData object in place.

    Parameters
    ----------
    adata : AnnData
        AnnData object to be normalized and log-transformed.

    Returns
    -------
    AnnData
        The same AnnData object after in-place modification.
    """
    print("Normalizing counts to 1e4 per cell...")
    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    print("Applying log1p transformation...")
    sc.pp.log1p(adata)
    return adata



def _digitize(x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
    """
    Assign values to discrete bins with optional randomized tie-breaking.

    Uses numpy digitization to map 1D input values to bin indices. When
    duplicate bin edges are present and `side="both"`, values are randomly
    assigned between left and right bin indices to avoid collapse.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional input array to be digitized.
    bins : np.ndarray
        One-dimensional array of bin edges in ascending order.
    side : str, optional
        Digitization strategy. If "one", uses left-side digitization only.
        If "both", randomly assigns between left and right digitization
        results. Defaults to "both".

    Returns
    -------
    np.ndarray
        Array of integer bin indices with the same length as `x`.
    """
    assert x.ndim == 1 and bins.ndim == 1

    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits

    right_difits = np.digitize(x, bins, right=True)

    rands = np.random.rand(len(x))  # uniform random numbers

    digits = rands * (right_difits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits


def binning(
    row: Union[np.ndarray, torch.Tensor], n_bins: int
) -> Union[np.ndarray, torch.Tensor]:
    """
    Bin a single row of non-negative values into discrete quantile-based bins.

    Supports both NumPy arrays and Torch tensors as input. Zero entries are
    preserved, and binning is applied only to non-zero values when present.
    The output type matches the input type.

    Parameters
    ----------
    row : np.ndarray or torch.Tensor
        One-dimensional input data to be binned.
    n_bins : int
        Number of bins to use for quantile-based discretization.

    Returns
    -------
    np.ndarray or torch.Tensor
        Binned representation of the input row, with the same shape and type
        as the input.
    """
    dtype = row.dtype
    return_np = False if isinstance(row, torch.Tensor) else True
    row = row.cpu().numpy() if isinstance(row, torch.Tensor) else row

    if row.max() == 0:
        print("The input data contains row of zeros. Please make sure this is expected.")
        
        return (
            np.zeros_like(row, dtype=dtype)
            if return_np
            else torch.zeros_like(row, dtype=dtype)
        )

    if row.min() <= 0:
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        non_zero_digits = _digitize(non_zero_row, bins)
        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
    else:
        bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
        binned_row = _digitize(row, bins)
    return torch.from_numpy(binned_row) if not return_np else binned_row.astype(dtype)


def do_binning(adata, n_bins):
    """
    Apply row-wise quantile binning to AnnData and store results as sparse matrices.

    Iterates over observations, bins non-zero values into discrete bins, and
    stores the binned data in `adata.layers`. Per-row bin edges are also stored
    in `adata.obsm`. Designed for large, sparse datasets with progress logging.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing non-negative input data.
    n_bins : int
        Number of bins to use for quantile-based discretization.

    Returns
    -------
    None
        Results are written to `adata.layers` and `adata.obsm` in place.
    """
    print("binning data ...")
    # custom
    key_to_process = None 
    result_binned_key = "X_binned"
    binned_sparse_rows = []
    bin_edges_sparse_rows = []

    layer_data = _get_obs_rep(adata, layer=key_to_process)
    n_rows = layer_data.shape[0]

    for i in range(n_rows):
        row = layer_data[i].toarray().ravel() if issparse(layer_data) else layer_data[i]
        if row.min() < 0:
            raise ValueError(
                f"Assuming non-negative data, but got min value {row.min()} in row {i}."
            )
        if row.max() == 0:
            print("The input data contains all zero rows. Please make sure this is expected.")
            binned_sparse_rows.append(csr_matrix(np.zeros_like(row, dtype=np.int64)))
            bin_edges_sparse_rows.append(csr_matrix([0] * n_bins))
            continue

        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        non_zero_digits = _digitize(non_zero_row, bins)

        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
        binned_sparse_rows.append(csr_matrix(binned_row))
        bin_edges_sparse_rows.append(csr_matrix(np.concatenate([[0], bins])))

        if i % 10000 == 0:
            print(f"[debug] processed {i + 1}/{n_rows} rows")


    binned_sparse = vstack(binned_sparse_rows)
    print('[debug] vstack')
    adata.layers[result_binned_key] = binned_sparse
    print('[debug] binned sparse matrix stored')

    bin_edges_sparse = vstack(bin_edges_sparse_rows)
    adata.obsm["bin_edges"] = bin_edges_sparse
    print('[debug] bin edges sparse matrix stored')

    print('[debug] binning done')




def do_gene_locating(adata, file='../../meta_info/gene_order.pkl'):
    """
    Map gene positions to AnnData based on an external gene order file.

    Parameters
    ----------
    adata : AnnData
        AnnData object whose variables correspond to gene names.
    file : str, optional
        Path to a pickle file containing gene metadata with `gene_name`
        and `pos` fields.

    Returns
    -------
    AnnData
        The same AnnData object with gene position information stored
        in `adata.layers`.
    """
    print("Locating the genes ...")

    result_binned_key = "gene_pos"

    with open(file, 'rb') as file:
        gene_data = pickle.load(file)

    gene_to_pos = {entry['gene_name']: entry['pos'] for entry in gene_data}
    

    num_cells = adata.X.shape[0]
    num_genes = adata.X.shape[1]

    gene_pos = np.zeros((num_cells, num_genes), dtype=int)


    missing_genes = set()  
    for i in range(num_cells):
        for j in range(num_genes):
            gene_name = adata.var_names[j]
            if gene_name in gene_to_pos:
                gene_pos[i, j] = gene_to_pos[gene_name] 
            else:
                missing_genes.add(gene_name) 

    adata.layers[result_binned_key] = gene_pos

    print(f"Total missing genes: {len(missing_genes)}")
    print(f"Missing genes: {list(missing_genes)}")
    return adata


def do_chr_locating(adata, file='../../meta_info/gene_order.pkl'):
    """
    Map chromosome indices to AnnData based on an external gene annotation file.

    Parameters
    ----------
    adata : AnnData
        AnnData object whose variables correspond to gene names.
    file : str, optional
        Path to a pickle file containing gene metadata with `gene_name`
        and `chr` fields.

    Returns
    -------
    AnnData
        The same AnnData object with chromosome information stored
        in `adata.layers`.
    """
    print("Locating the chromosomes ...")

    key_to_process = None 
    result_binned_key = "gene_chr"

    with open(file, 'rb') as file:
        gene_data = pickle.load(file)

    gene_to_chr = {entry['gene_name']: entry['chr'] for entry in gene_data}
    num_cells = adata.X.shape[0]
    num_genes = adata.X.shape[1]

    gene_chr = np.zeros((num_cells, num_genes), dtype=int)

  
    missing_genes = set()  
    for i in range(num_cells):
        for j in range(num_genes):
            gene_name = adata.var_names[j] 
            if gene_name in gene_to_chr:
                gene_chr[i, j] = gene_to_chr[gene_name]  
            else:
                missing_genes.add(gene_name)

    adata.layers[result_binned_key] = gene_chr

    print(f"Total missing genes: {len(missing_genes)}")
    print(f"Missing genes: {list(missing_genes)}")
    return adata

def preprocess_h5ad(h5ad_path, do_norm_and_log1p=False):
    """
    Load and preprocess an h5ad file for downstream analysis.

    Parameters
    ----------
    h5ad_path : str
        Path to the input `.h5ad` file.
    do_norm_and_log1p : bool, optional
        Whether to apply total-count normalization and log1p
        transformation before binning. Defaults to False.

    Returns
    -------
    AnnData
        Preprocessed AnnData object.
    """
    adata = sc.read_h5ad(h5ad_path)
    if do_norm_and_log1p == True:
        do_normalizing_and_log1p(adata)
    do_binning(adata, 51) 
    return adata
