from typing import Any, Callable, List, Optional, Union

import torch
import torch.nn.functional as F
from deepspeed.moe.layer import MoE
from torch import Tensor, nn


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
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)
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


class Cls2ExpValueDecoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x: Tensor):
        return self.decoder(x)


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
        dim_feedforward: int = 128,
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

        out, attn_weights = self._sa_block(src, src_mask, src_key_padding_mask, is_causal=is_causal)
        x = self.norm1(src + out)
        x_cls = x[:, 0, :]
        y, l_aux, exp_count, custom_output, combine_weights, original_logits = self._ff_block(x)
        x = self.norm2(y + x)

        batchsize, seqlen, embsize1 = x.shape
        e, bsl, embsize2 = custom_output.shape
        assert batchsize * seqlen == bsl
        assert embsize1 == embsize2

        custom_output = custom_output.reshape(e, batchsize, seqlen, embsize2)
        custom_output = custom_output[:, :, 0, :]
        custom_output = self.norm2(custom_output + x_cls)

        combine_weights = combine_weights.reshape(batchsize, seqlen, e, -1)
        gate_cls = combine_weights[:, 0, :, :].sum(dim=-1)

        original_logits = original_logits.reshape(batchsize, seqlen, e)
        gate_logits_cls = original_logits[:, 0, :]

        return x, l_aux, exp_count, custom_output, gate_cls, gate_logits_cls, attn_weights

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False):
        attn_output, attn_weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        return self.dropout1(attn_output), attn_weights

    def _ff_block(self, x: Tensor):
        moe_out, l_aux, exp_count, custom_output, gate_weight, original_logits = self.moe_ffn(x)
        return self.dropout2(moe_out), l_aux, exp_count, custom_output, gate_weight, original_logits


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
        moe_experts: int = 6,
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
        gate_logits_cls = None
        attn_weights = None
        attn_weights_all_layers = []
        for layer in self.layers:
            if isinstance(layer, MoETransformerEncoderLayer):
                x, l_aux, exp_count, custom_output, gate_weight_cls, gate_logits_cls, attn_weights = layer(
                    x, src_mask, src_key_padding_mask, is_causal
                )
                aux_losses.append(l_aux)
                expert_counts.append(exp_count)
                attn_weights_all_layers.append(attn_weights)
            else:
                x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)
        return x, aux_losses, expert_counts, custom_output, gate_weight_cls, gate_logits_cls, attn_weights, attn_weights_all_layers


class TransformerModel(nn.Module):
    def __init__(
        self,
        ntokens: int,
        num_layers: int,
        d_model: int,
        nhead: int,
        seq_len: Optional[int],
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
        mlm: bool = True,
        dim_feedforward: int = 1024,
        aux_lambda: float = 0.8,
        moe_experts: int = 6,
        epsize: int = 1,
        top_k: int = 1,
        moe_layers: List[int] = [3, 5, 7, 9, 11],
    ) -> None:
        super().__init__()
        del moe, seq_len
        self.mlm = mlm
        self.aux_lambda = aux_lambda

        self.encoder1 = GeneNameEncoder(num_embeddings=ntokens + 1, embedding_dim=d_model, padding_idx=vocab[pad_token])
        self.encoder2 = GeneExpValueEncoder(embedding_dim=d_model)
        self.encoder3 = GeneChrEncoder(num_embeddings=27, embedding_dim=d_model, padding_idx=26)
        self.encoder4 = GenePosEncoder()
        self.transformer_enc = MoETransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            moe_experts=moe_experts,
            epsize=epsize,
            top_k=top_k,
            moe_layers=moe_layers,
        )
        self.decoder1 = GeneExpValueDecoder(embedding_dim=d_model)
        self.decoder3 = Cls2ExpValueDecoder(embedding_dim=d_model)

    def masked_mse_loss(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float()
        loss = F.mse_loss(input * mask, target * mask, reduction="sum")
        return loss / mask.sum()

    def forward(self, return_cell_emb=False, inf=False, **inputs):
        del return_cell_emb
        gene_id = inputs["gene_id"]
        pos = inputs["pos"]
        chr = inputs["chr"]
        src_key_padding_mask = inputs["src_key_padding_mask"]
        target_exp = inputs["target_exp"]

        if not inf:
            masked_exp = inputs["masked_exp"].squeeze(1)
            exp_mask_bool = inputs["exp_mask_bool"].squeeze(1)
            exp_value_enc = self.encoder2(masked_exp)
        else:
            exp_value_enc = self.encoder2(target_exp)
            exp_mask_bool = None

        gene_ids_enc = self.encoder1(gene_id)
        chr_enc = self.encoder3(chr)
        pos_enc = self.encoder4(exp_value_enc, pos)
        chr_enc = chr_enc.to(dtype=next(self.parameters()).dtype)
        pos_enc = pos_enc.to(dtype=next(self.parameters()).dtype)
        total_enc = gene_ids_enc + exp_value_enc + chr_enc + pos_enc

        output1, aux_losses, expert_counts, custom_outputs, gate_weight_cls, gate_logits_cls, attn_weights, attn_weights_all_layers = self.transformer_enc(
            total_enc,
            src_key_padding_mask=src_key_padding_mask,
        )
        del expert_counts, custom_outputs, gate_logits_cls, attn_weights_all_layers

        cell_emb = output1[:, 0, :]
        l_aux_total = sum(aux_losses) / len(aux_losses)

        if not inf:
            output2 = self.decoder1(output1)
            mse_loss = self.masked_mse_loss(output2["pred"], target_exp, exp_mask_bool)
            output3 = self.decoder3(cell_emb)
            output1_t = output1.transpose(1, 2)
            pred_exp_bycls = torch.einsum("bd,bdn->bn", output3, output1_t)
            mse_loss_bycls = self.masked_mse_loss(pred_exp_bycls, target_exp, exp_mask_bool)
            total_loss = mse_loss + mse_loss_bycls + self.aux_lambda * l_aux_total
            l_aux_value = l_aux_total.item() if isinstance(l_aux_total, torch.Tensor) else float(l_aux_total)
            return {
                "total_loss": total_loss,
                "mse_loss": mse_loss,
                "mse_loss_bycls": mse_loss_bycls,
                "l_aux": l_aux_value,
                "aux_lambda": self.aux_lambda,
            }

        attn_weights_celltype = attn_weights.mean(dim=0)
        l_aux_value = l_aux_total.item() if isinstance(l_aux_total, torch.Tensor) else float(l_aux_total)
        return {
            "l_aux": l_aux_value,
            "aux_lambda": self.aux_lambda,
            "target_exp": target_exp,
            "attn_weights_celltype": attn_weights_celltype.detach().cpu(),
            "gate_weight_cls": gate_weight_cls.detach().cpu(),
        }
