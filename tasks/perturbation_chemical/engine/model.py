from typing import Any, Callable, List, Optional, Union

import torch
import torch.nn.functional as F
from rdkit import Chem
from torch import Tensor, nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool

from deepspeed.moe.layer import MoE


class GeneNameEncoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.enc_norm(self.embedding(x))


class GeneExpValueEncoder(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(embedding_dim)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-1)
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class GeneChrEncoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.enc_norm(self.embedding(x))


class GenePosEncoder(nn.Module):
    def __init__(self, div_term_scale=1000):
        super().__init__()
        self.div_term_scale = div_term_scale

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        x = x.squeeze(1)
        batchsize, seq_len, embsize = x.size()
        position_encoding = torch.zeros(batchsize, seq_len, embsize, device=x.device)
        div_term = torch.pow(10000, torch.arange(0.0, embsize, 2, device=x.device) / embsize) / self.div_term_scale
        for i in range(embsize // 2):
            position_encoding[:, :, 2 * i] = torch.sin(positions * div_term[i])
            position_encoding[:, :, 2 * i + 1] = torch.cos(positions * div_term[i])
        return position_encoding


class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, gcn_output_dim, emb_size, seq_len):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, gcn_output_dim)
        self.linear = nn.Linear(gcn_output_dim, emb_size)
        self.seq_len = seq_len

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        graph_emb = global_mean_pool(x, batch)
        graph_emb = self.linear(graph_emb)
        return graph_emb.unsqueeze(1).expand(-1, self.seq_len, -1)


class GeneExpValueDecoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, x: Tensor):
        return {"pred": self.fc(x).squeeze(-1)}


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    edge_index = [[], []]
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edge_index[0] += [start, end]
        edge_index[1] += [end, start]

    return Data(
        x=torch.tensor(atom_features, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )


def batch_smiles_to_batch_graph(smiles_list):
    data_list = []
    for smi in smiles_list:
        data = smiles_to_graph(smi)
        if data is not None:
            data_list.append(data)
    return Batch.from_data_list(data_list)


class ExpertModel(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


class MoETransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
        moe_experts: int = 6,
        epsize: int = 1,
        top_k: int = 2,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )
        self.moe_ffn = MoE(
            hidden_size=d_model,
            expert=ExpertModel(d_model, dim_feedforward, dropout),
            num_experts=moe_experts,
            ep_size=epsize,
            k=top_k,
        )
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        self.activation = activation

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, is_causal: bool = False):
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )
        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            y, l_aux, exp_count, custom_output = self._ff_block(self.norm2(x))
            x = x + y
            gate_cls = None
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x_cls = x[:, 0, :]
            y, l_aux, exp_count, custom_output, combine_weights = self._ff_block(x)
            x = self.norm2(y + x)
            batchsize, seqlen, embsize1 = x.shape
            e, bsl, embsize2 = custom_output.shape
            assert batchsize * seqlen == bsl
            assert embsize1 == embsize2
            custom_output = custom_output.reshape(e, batchsize, seqlen, embsize2)
            custom_output = custom_output[:, :, 0, :]
            custom_output = self.norm2(custom_output + x_cls)
            combine_weights = combine_weights.reshape(batchsize, seqlen, e, -1)
            gate_cls = combine_weights[:, 0, :, :].mean(dim=-1)

        return x, l_aux, exp_count, custom_output, gate_cls

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor):
        moe_out, l_aux, exp_count, custom_output, gate_weight, _ = self.moe_ffn(x)
        return self.dropout2(moe_out), l_aux, exp_count, custom_output, gate_weight


class MoETransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
        moe_experts: int = 2,
        epsize: int = 1,
        top_k: int = 2,
        moe_layers: List[int] = [3, 5, 7, 9, 11],
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i in moe_layers:
                self.layers.append(
                    MoETransformerEncoderLayer(
                        d_model,
                        nhead,
                        dim_feedforward,
                        dropout,
                        activation,
                        layer_norm_eps,
                        batch_first,
                        norm_first,
                        bias,
                        device,
                        dtype,
                        moe_experts,
                        epsize,
                        top_k,
                    )
                )
            else:
                self.layers.append(
                    nn.TransformerEncoderLayer(
                        d_model,
                        nhead,
                        dim_feedforward,
                        dropout,
                        activation=activation,
                        layer_norm_eps=layer_norm_eps,
                        batch_first=batch_first,
                        norm_first=norm_first,
                        bias=bias,
                        device=device,
                        dtype=dtype,
                    )
                )

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, is_causal: bool = False):
        x = src
        aux_losses = []
        expert_counts = []
        custom_output = None
        gate_weight_cls = None
        for layer in self.layers:
            if isinstance(layer, MoETransformerEncoderLayer):
                x, l_aux, exp_count, custom_output, gate_weight_cls = layer(x, src_mask, src_key_padding_mask, is_causal)
                aux_losses.append(l_aux)
                expert_counts.append(exp_count)
            else:
                x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)
        return x, aux_losses, expert_counts, custom_output, gate_weight_cls


class TransformerModel(nn.Module):
    def __init__(
        self,
        ntokens: int,
        num_layers: int,
        d_model: int,
        nhead: int,
        seq_len: int = 1601,
        pad_token: str = "<pad>",
        vocab: Any = None,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
        moe=True,
        mlm: bool = False,
        cls: bool = False,
        pert_drug: bool = True,
        dim_feedforward: int = 1024,
        aux_lambda: float = 0.15,
        moe_experts: int = 6,
        epsize: int = 1,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.mlm = mlm
        self.cls = cls
        self.pert_drug = pert_drug
        self.initial_lambda = aux_lambda
        self.aux_lambda = aux_lambda

        self.encoder1 = GeneNameEncoder(num_embeddings=ntokens + 1, embedding_dim=d_model, padding_idx=vocab[pad_token])
        self.encoder2 = GeneExpValueEncoder(embedding_dim=d_model)
        self.encoder3 = GeneChrEncoder(num_embeddings=27, embedding_dim=d_model, padding_idx=26)
        self.encoder4 = GenePosEncoder()
        self.encoder6 = GNNEncoder(input_dim=1, hidden_dim=64, gcn_output_dim=128, emb_size=d_model, seq_len=seq_len)

        self.transformer_enc = MoETransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            moe_experts=moe_experts,
            epsize=epsize,
            top_k=top_k,
        )

        if mlm or pert_drug:
            self.decoder1 = GeneExpValueDecoder(embedding_dim=d_model)

    def masked_mse_loss(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float()
        loss = F.mse_loss(input * mask, target * mask, reduction="sum")
        return loss / mask.sum()

    def compute_pert_drug_metrics(self, ctrl_exp, target_exp, pred_exp, mask_mode: str = "delta", threshold: float = 0.1):
        metrics = {}
        if mask_mode == "delta":
            mask = torch.abs(target_exp - ctrl_exp) > threshold
        elif mask_mode == "target":
            mask = target_exp > threshold
        elif mask_mode == "ctrl":
            mask = ctrl_exp > threshold
        elif mask_mode == "none":
            mask = torch.ones_like(target_exp).bool()
        else:
            raise ValueError(f"Unsupported mask_mode: {mask_mode}")

        pred_masked = pred_exp[mask]
        target_masked = target_exp[mask]
        metrics["mse"] = F.mse_loss(pred_masked, target_masked)
        l2_per_sample = torch.norm((pred_exp - target_exp) * mask.float(), p=2, dim=1)
        metrics["l2"] = l2_per_sample.mean()
        metrics["pearson"] = self.batchwise_pearson(pred_exp, target_exp)
        metrics["delta_pearson"] = self.batchwise_pearson(pred_exp - ctrl_exp, target_exp - ctrl_exp)
        metrics["spearman"] = self.batchwise_spearman(pred_exp, target_exp)
        metrics["delta_spearman"] = self.batchwise_spearman(pred_exp - ctrl_exp, target_exp - ctrl_exp)
        return metrics

    def batchwise_pearson(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)
        x_centered = x - x_mean
        y_centered = y - y_mean
        numerator = (x_centered * y_centered).sum(dim=1)
        denominator = x_centered.pow(2).sum(dim=1).sqrt() * y_centered.pow(2).sum(dim=1).sqrt()
        pearson = numerator / (denominator + 1e-8)
        valid_mask = ~torch.isnan(pearson)
        if valid_mask.sum() == 0:
            return torch.tensor(float("nan"), device=x.device)
        return pearson[valid_mask].mean()

    def batchwise_spearman(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        def rank(t: torch.Tensor) -> torch.Tensor:
            return t.argsort(dim=1).argsort(dim=1).float()

        return self.batchwise_pearson(rank(x), rank(y))

    def forward(self, return_cell_emb=False, inf=False, **inputs):
        del return_cell_emb
        gene_id = inputs["gene_id"]
        pos = inputs["pos"]
        chr = inputs["chr"]
        src_key_padding_mask = inputs["src_key_padding_mask"]
        target_exp = inputs["target_exp"]
        ctrl_exp = inputs["ctrl_exp"]
        smiles = batch_smiles_to_batch_graph(inputs["smiles"])
        sample = inputs["sample"]
        del sample

        device = ctrl_exp.device
        smiles = smiles.to(device)

        if self.mlm:
            masked_exp = inputs["masked_exp"].squeeze(1)
            exp_mask_bool = inputs["exp_mask_bool"].squeeze(1)

        gene_ids_enc = self.encoder1(gene_id)
        c_exp_value_enc = self.encoder2(ctrl_exp)
        chr_enc = self.encoder3(chr)
        pos_enc = self.encoder4(c_exp_value_enc, pos)
        smiles_enc = self.encoder6(smiles)

        chr_enc = chr_enc.to(dtype=next(self.parameters()).dtype)
        pos_enc = pos_enc.to(dtype=next(self.parameters()).dtype)
        total_enc = gene_ids_enc + c_exp_value_enc + chr_enc + pos_enc + smiles_enc

        output1, aux_losses, expert_counts, custom_outputs, gate_weight_cls = self.transformer_enc(
            total_enc, src_key_padding_mask=src_key_padding_mask
        )
        del expert_counts, custom_outputs, gate_weight_cls
        l_aux_total = sum(aux_losses) / len(aux_losses)

        if self.mlm:
            output2 = self.decoder1(output1)
            mse_loss = self.masked_mse_loss(output2["pred"], target_exp, exp_mask_bool)
            total_loss = mse_loss + self.aux_lambda * l_aux_total

        if self.pert_drug:
            output2 = self.decoder1(output1)
            metrics = self.compute_pert_drug_metrics(ctrl_exp, target_exp, output2["pred"], mask_mode="none")
            l2_loss = metrics["l2"]
            total_loss = l2_loss + self.aux_lambda * l_aux_total

        if isinstance(l_aux_total, torch.Tensor):
            l_aux_total = l_aux_total.item()
        else:
            l_aux_total = float(l_aux_total)

        output = {
            "total_loss": total_loss,
            "l2_loss": metrics["l2"].item(),
            "mse": metrics["mse"].item(),
            "pearson": metrics["pearson"].item(),
            "delta_pearson": metrics["delta_pearson"].item(),
            "spearman": metrics["spearman"].item(),
            "delta_spearman": metrics["delta_spearman"].item(),
            "l_aux": l_aux_total.item() if hasattr(l_aux_total, "item") else l_aux_total,
            "aux_lambda": self.aux_lambda.item() if hasattr(self.aux_lambda, "item") else self.aux_lambda,
        }

        if inf:
            output["pred_exp"] = output2["pred"]

        return output
