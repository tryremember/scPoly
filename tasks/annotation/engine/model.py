from typing import Optional, Any, Union, Callable,List
import os
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from deepspeed.moe.layer import MoE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.autograd import Function
from typing import Optional

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
print(torch.__version__)  
print(torch.version.cuda)  
print(torch.cuda.is_available()) 
print(torch.cuda.get_device_name(0)) 
print(torch.cuda.get_device_capability(0))
torch.cuda.empty_cache()


class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification.

    Extends cross-entropy loss by down-weighting well-classified samples
    and focusing training on hard examples. Optional per-class weighting
    is supported via `alpha`.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Initialize the focal loss module.

        Parameters
        ----------
        gamma : float, optional
            Focusing parameter controlling the strength of modulation.
        alpha : torch.Tensor or None, optional
            Per-class weight tensor. If provided, it is indexed by target labels.
        reduction : str, optional
            Reduction method applied to the output loss.
            Supported values: "mean", "sum", or "none".
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Compute focal loss from logits and targets.

        Parameters
        ----------
        logits : torch.Tensor
            Raw prediction logits of shape (batch_size, num_classes).
        targets : torch.Tensor
            Ground-truth class indices of shape (batch_size,).

        Returns
        -------
        torch.Tensor
            Reduced loss value if reduction is "mean" or "sum",
            otherwise per-sample loss tensor.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            at = self.alpha.to(targets.device)
            at = at[targets]
            focal_loss = at * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class ExpertModel(nn.Module):
    """
    Feed-forward expert module used inside a Mixture-of-Experts (MoE) layer.
    """
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        """
        Initialize the expert feed-forward network.

        Parameters
        ----------
        d_model : int
            Input and output feature dimension.
        dim_feedforward : int
            Hidden layer dimension of the expert.
        dropout : float, optional
            Dropout probability applied between linear layers.
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the expert feed-forward transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., d_model).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., d_model).
        """
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    """
    Return an activation function by name.

    Parameters
    ----------
    activation : str
        Name of the activation function. Supported values are
        "relu" and "gelu".

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        Activation function.

    Raises
    ------
    RuntimeError
        If the activation name is not supported.
    """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


class MoETransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with a Mixture-of-Experts (MoE) feed-forward block.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        bias: bool = True,
        *,
        moe_experts: int,
        epsize: int,
        top_k: int,
        device=None,
        dtype=None,
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

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        self.activation = activation

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ):
        """
        Apply a MoE-based Transformer encoder layer.

        Returns
        -------
        Tuple
            - Output tensor (batch, seq_len, d_model)
            - MoE auxiliary loss
            - Expert usage counts
            - Per-expert CLS-level outputs
            - Gating weights for CLS token
        """
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

        # Self-attention
        x = self.norm1(
            src + self._sa_block(src, src_mask, src_key_padding_mask, is_causal)
        )

        x_cls = x[:, 0, :]

        # MoE feed-forward
        y, l_aux, exp_count, custom_output, combine_weights = self._ff_block(x)
        x = self.norm2(x + y)

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

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
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
    """
    Stacked Transformer encoder with optional MoE layers.

    Builds a sequence of encoder layers where selected layer indices use a
    custom MoE-based encoder layer and the remaining layers use the standard
    PyTorch TransformerEncoderLayer.
    """
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
        bias: bool = True,
        *,
        moe_experts: int,
        epsize: int,
        top_k: int,
        moe_layers: List[int],
        device=None,
        dtype=None,
    ) -> None:
        """
        Initialize the encoder stack.

        Parameters
        ----------
        num_layers : int
            Total number of encoder layers.
        d_model : int
            Model embedding dimension.
        nhead : int
            Number of attention heads.
        dim_feedforward : int
            Feed-forward hidden dimension.
        dropout : float, optional
            Dropout probability.
        activation : str or callable, optional
            Activation function for encoder layers.
        layer_norm_eps : float, optional
            Epsilon for layer normalization.
        batch_first : bool, optional
            If True, input/output tensors are (batch, seq, feature).
        bias : bool, optional
            Whether to use bias terms in attention/projections.
        device : optional
            Module device.
        dtype : optional
            Module dtype.
        moe_experts : int, optional
            Number of experts in MoE layers.
        epsize : int, optional
            Expert parallel size passed to MoE.
        top_k : int, optional
            Number of experts selected per token.
        moe_layers : List[int], optional
            Layer indices that should use MoETransformerEncoderLayer.
        """
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i in moe_layers:
                self.layers.append(
                    MoETransformerEncoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        activation=activation,
                        layer_norm_eps=layer_norm_eps,
                        batch_first=batch_first,
                        bias=bias,
                        moe_experts=moe_experts,
                        epsize=epsize,
                        top_k=top_k,
                        device=device,
                        dtype=dtype,
                    )
                )
            else:
                self.layers.append(
                    nn.TransformerEncoderLayer(
                        d_model, nhead, dim_feedforward, dropout,
                        activation=activation, layer_norm_eps=layer_norm_eps,
                        batch_first=batch_first, bias=bias,
                        device=device, dtype=dtype
                    )
                )

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """
        Apply the encoder stack.

        Parameters
        ----------
        src : torch.Tensor
            Input tensor of shape (batch, seq_len, d_model) when batch_first=True.
        src_mask : torch.Tensor, optional
            Attention mask.
        src_key_padding_mask : torch.Tensor, optional
            Padding mask for attention.
        is_causal : bool, optional
            Whether to apply causal masking.

        Returns
        -------
        Tuple
            - Final hidden states
            - List of auxiliary MoE losses (one per MoE layer encountered)
            - List of expert usage counts (one per MoE layer encountered)
            - Per-expert outputs from the last MoE layer encountered
            - CLS gating weights from the last MoE layer encountered
        """
        x = src
        aux_losses = []
        expert_counts = []

        for layer in self.layers:
            if isinstance(layer, MoETransformerEncoderLayer):
                x, l_aux, exp_count, custom_output, gate_weight_cls = layer(
                    x, src_mask, src_key_padding_mask, is_causal
                )
                aux_losses.append(l_aux)
                expert_counts.append(exp_count)
            else:
                x = layer(
                    x,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    is_causal=is_causal,
                )

        return x, aux_losses, expert_counts, custom_output, gate_weight_cls


class TransformerModel(nn.Module):
    """
    Main model.

    The model composes:
    - gene ID embeddings
    - expression value embeddings
    - chromosome embeddings
    - sinusoidal position encodings
    - a stacked Transformer encoder with optional MoE layers
    - optional heads for classification, masked value prediction, and batch discrimination (GRL)
    """

    def __init__(
        self,
        ntokens: int,
        num_layers: int,
        d_model: int,
        nhead: int,
        n_cls: int,
        n_batch: int = None,
        pad_token: str = "<pad>",
        vocab: Any = None,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
        GRL: bool = False,
        mlm: bool = False,
        cls: bool = True,
        alpha: torch.Tensor = None,
        dim_feedforward: int = 2048,
        aux_lambda: float = 0.15,
        beta: float = 1,
        moe_experts: int = 6,
        epsize: int = 1,
        top_k: int = 2,
        moe_layers: List[int] = [3, 5, 7, 9, 11],
    ) -> None:
        """
        Initialize the model.

        Parameters
        ----------
        ntokens : int
            Vocabulary size (gene tokens).
        num_layers : int
            Number of encoder layers.
        d_model : int
            Model embedding dimension.
        nhead : int
            Number of attention heads.
        n_cls : int
            Number of classification targets.
        n_batch : int
            Number of batch classes for discriminator.
        pad_token : str, optional
            Padding token string used to resolve padding index from vocab.
        vocab : Any, optional
            Vocabulary object supporting vocab[pad_token] -> pad index.
        dropout : float, optional
            Dropout probability.
        activation : str or callable, optional
            Activation function used by transformer layers.
        layer_norm_eps : float, optional
            LayerNorm epsilon.
        batch_first : bool, optional
            If True, tensors are (batch, seq, feature).
        bias : bool, optional
            Whether to use bias terms in transformer layers.
        device, dtype : optional
            Module device/dtype.
        GRL : bool, optional
            Enable batch discriminator head.
        mlm : bool, optional
            Enable masked value prediction head.
        cls : bool, optional
            Enable classification head.
        alpha : torch.Tensor, optional
            Class weights for focal loss.
        dim_feedforward : int, optional
            Feed-forward hidden dimension in transformer layers.
        aux_lambda : float, optional
            Weight for MoE auxiliary loss term (MLM path).
        beta : float, optional
            Weight for discriminator loss.
        moe_experts : int, optional
            Number of experts in MoE layers.
        epsize : int, optional
            Expert parallel size passed to MoE.
        top_k : int, optional
            Number of experts selected per token.
        moe_layers : List[int], optional
            Layer indices that should use MoETransformerEncoderLayer.
        """
        super().__init__()

        self.mlm = mlm
        self.cls = cls
        self.GRL = GRL

        self.alpha = alpha
        self.initial_lambda = aux_lambda
        self.aux_lambda = aux_lambda
        self.beta = beta

        self.encoder1 = GeneNameEncoder(
            num_embeddings=ntokens + 1, embedding_dim=d_model, padding_idx=vocab[pad_token]
        )
        self.encoder2 = GeneExpValueEncoder(embedding_dim=d_model)
        self.encoder3 = GeneChrEncoder(num_embeddings=27, embedding_dim=d_model, padding_idx=26)
        self.encoder4 = GenePosEncoder()

        self.transformer_enc = MoETransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            bias=bias,
            device=device,
            dtype=dtype,
            moe_experts=moe_experts,
            epsize=epsize,
            top_k=top_k,
            moe_layers=moe_layers,
        )

        if self.GRL:
            self.decoder4 = BatchDecoder(embedding_dim=d_model, n_batch=n_batch)
        if mlm:
            self.decoder1 = GeneExpValueDecoder(embedding_dim=d_model)
        if cls:
            self.decoder2 = ClsDecoder(embedding_dim=d_model, n_cls=n_cls)

    def masked_mse_loss(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE on masked positions only.

        Parameters
        ----------
        input : torch.Tensor
            Predicted values.
        target : torch.Tensor
            Target values.
        mask : torch.Tensor
            Boolean or 0/1 mask tensor.

        Returns
        -------
        torch.Tensor
            Scalar loss normalized by number of masked positions.
        """
        mask = mask.float()
        loss = F.mse_loss(input * mask, target * mask, reduction="sum")
        return loss / mask.sum()

    def cls_loss(
        self,
        output_cls: torch.Tensor,
        target_cls: torch.Tensor,
        use_focal: bool = False,
        gamma: float = 2.0,
        alpha: torch.Tensor = None,
    ):
        """
        Compute classification loss.

        Parameters
        ----------
        output_cls : torch.Tensor
            Logits of shape (batch_size, n_cls).
        target_cls : torch.Tensor
            Target indices of shape (batch_size,).
        use_focal : bool, optional
            If True, uses focal loss.
        gamma : float, optional
            Focal loss gamma.
        alpha : torch.Tensor, optional
            Per-class weights for focal loss.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        if use_focal:
            if alpha is None:
                alpha = self.alpha
            criterion = FocalLoss(gamma=gamma, alpha=alpha)
        else:
            criterion = nn.CrossEntropyLoss()

        loss = criterion(output_cls, target_cls)
        return loss

    def discriminator_loss(self, output_dis: torch.Tensor, target_dis: torch.Tensor) -> torch.Tensor:
        """
        Compute discriminator loss and accuracy for batch prediction.

        Returns
        -------
        Tuple[torch.Tensor, float]
            Cross-entropy loss and accuracy.
        """
        criterion_dis = nn.CrossEntropyLoss()
        loss = criterion_dis(output_dis, target_dis)
        preds = torch.argmax(output_dis, dim=1)
        correct = (preds == target_dis).sum().item()
        total = target_dis.size(0)
        acc = correct / total
        return loss, acc

    def compute_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute classification accuracy.

        Parameters
        ----------
        logits : torch.Tensor
            Logits of shape (batch_size, num_classes).
        targets : torch.Tensor
            Target indices of shape (batch_size,).

        Returns
        -------
        float
            Accuracy in [0, 1].
        """
        preds = torch.argmax(logits, dim=1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)
        return correct / total

    def compute_metrics(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Compute macro metrics for multi-class classification.

        Returns
        -------
        Tuple[float, float, float, float]
            (accuracy, precision, recall, f1)
        """
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        labels = targets.cpu().numpy()
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="macro", zero_division=0)
        recall = recall_score(labels, preds, average="macro", zero_division=0)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)

        return accuracy, precision, recall, f1

    def forward(self, return_cell_emb=False, inf=False, **inputs):
        """
        Forward pass.

        inf=False:
            Requires batch_id/celltype_id and returns loss + metrics.
        inf=True:
            Skips label losses and returns logits/embeddings for inference.
        """
        gene_id = inputs["gene_id"]
        pos = inputs["pos"]
        chr = inputs["chr"]
        src_key_padding_mask = inputs["src_key_padding_mask"]
        target_exp = inputs["target_exp"]

        batch_id = inputs.get("batch_id", None)
        celltype_id = inputs.get("celltype_id", None)

        if self.mlm:
            masked_exp = inputs["masked_exp"]
            exp_mask_bool = inputs["exp_mask_bool"]
            masked_exp = masked_exp.squeeze(1)
            exp_mask_bool = exp_mask_bool.squeeze(1)

        # Encoders
        gene_ids_enc = self.encoder1(gene_id)
        exp_value_enc = self.encoder2(target_exp)
        chr_enc = self.encoder3(chr)
        pos_enc = self.encoder4(exp_value_enc, pos)

        chr_enc = chr_enc.to(dtype=next(self.parameters()).dtype)
        pos_enc = pos_enc.to(dtype=next(self.parameters()).dtype)

        total_enc = gene_ids_enc + exp_value_enc + chr_enc + pos_enc

        # Transformer
        output1, aux_losses, expert_counts, custom_outputs, gate_weight_cls = self.transformer_enc(
            total_enc, src_key_padding_mask=src_key_padding_mask
        )

        cell_emb = output1[:, 0, :]
        cell_emb_n_expert = custom_outputs

        l_aux_total = sum(aux_losses) / len(aux_losses)

        # Normalize l_aux_total to a Python float for logging/return
        if isinstance(l_aux_total, torch.Tensor):
            l_aux_scalar = l_aux_total.item()
        else:
            l_aux_scalar = float(l_aux_total)

        total_loss = None
        mse_loss = None
        cls_loss = None
        discriminator_loss = None
        discriminator_acc = None
        cls_pred = None
        accuracy = None
        precision = None
        recall = None
        f1 = None

        # MLM head
        if self.mlm:
            output2 = self.decoder1(output1)
            mse_loss = self.masked_mse_loss(output2["pred"], target_exp, exp_mask_bool)
            total_loss = mse_loss + self.aux_lambda * l_aux_total

        # CLS head
        if self.cls:
            cls_pred = self.decoder2(cell_emb)

            if not inf:
                cls_target = celltype_id
                cls_loss = self.cls_loss(cls_pred, cls_target, use_focal=False)
                total_loss = cls_loss + (l_aux_total - 1) ** 2
                accuracy, precision, recall, f1 = self.compute_metrics(cls_pred, cls_target)

        # GRL discriminator head (train only)
        if self.GRL and (not inf):
            discriminator_target = batch_id
            discriminator_pred = self.decoder4(cell_emb)
            discriminator_loss, discriminator_acc = self.discriminator_loss(
                discriminator_pred, discriminator_target
            )
            total_loss += self.beta * discriminator_loss


        if not inf:
            if self.cls == True and self.GRL == False:
                return {
                    "cls_loss": cls_loss,
                    "total_loss": total_loss,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "l_aux": l_aux_scalar,
                    "aux_lambda": self.aux_lambda.item() if hasattr(self.aux_lambda, "item") else self.aux_lambda,
                    "logits": cls_pred,
                    "cell_emb": cell_emb.detach().cpu(),
                    "labels": celltype_id.detach().cpu(),
                }
            elif self.cls == True and self.GRL == True:
                return {
                    "cls_loss": cls_loss,
                    "total_loss": total_loss,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "discriminator_acc": discriminator_acc,
                    "discriminator_loss": discriminator_loss,
                    "l_aux": l_aux_scalar,
                    "aux_lambda": self.aux_lambda.item() if hasattr(self.aux_lambda, "item") else self.aux_lambda,
                    "logits": cls_pred,
                    "cell_emb": cell_emb.detach().cpu(),
                    "labels": celltype_id.detach().cpu(),
                }
            else:
                return

        else:
            if self.cls == True:
                return {
                    "l_aux": l_aux_scalar,
                    "aux_lambda": self.aux_lambda.item() if hasattr(self.aux_lambda, "item") else self.aux_lambda,
                    "gate_weight_cls": gate_weight_cls,
                    "logits": cls_pred,
                    "cell_emb": cell_emb.detach().cpu(),
                    "expert_counts": expert_counts,
                    "cell_emb_n_expert": cell_emb_n_expert.detach().cpu(),
                }
            else:
                return


class GradientReversal(Function):
    """
    Gradient Reversal Layer (GRL) implementation.
    """

    @staticmethod
    def forward(ctx, x, alpha=1.0):
        """
        Forward pass of the gradient reversal layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        alpha : float, optional
            Scaling factor applied to gradients during backpropagation.

        Returns
        -------
        torch.Tensor
            Output tensor with the same value and shape as input.
        """
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the gradient reversal layer.

        Parameters
        ----------
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output.

        Returns
        -------
        Tuple[torch.Tensor, None]
            Reversed and scaled gradient for the input tensor, and
            None for the alpha argument.
        """
        return -ctx.alpha * grad_output, None

def grad_reverse(x, alpha=1.0):
    """
    Apply gradient reversal to the input tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    alpha : float, optional
        Scaling factor applied to gradients during backpropagation.

    Returns
    -------
    torch.Tensor
        Tensor with gradient reversal applied during backward pass.
    """
    return GradientReversal.apply(x, alpha)


class GeneExpValueDecoder(nn.Module):
    """
    Decode embedding representations into scalar expression value predictions.
    """
    def __init__(
        self,
        embedding_dim: int,
    ):
        """
        Initialize the expression value decoder.

        Parameters
        ----------
        embedding_dim : int
            Dimensionality of the input embedding.
        """
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, x: Tensor):
        """
        Decode embeddings into predicted expression values.

        Parameters
        ----------
        x : torch.Tensor
            Input embedding tensor of shape (batch_size, seq_len, embedding_dim).

        Returns
        -------
        dict
            Dictionary containing:
            - "pred": predicted expression values of shape (batch_size, seq_len).
        """
        pred_value = self.fc(x).squeeze(-1)
        return dict(pred=pred_value)


class ClsDecoder(nn.Module):
    """
    Classification head for predicting class labels from embeddings.
    """
    def __init__(
        self,
        embedding_dim: int,
        n_cls: int,
        dropout: float = 0.1
    ):
        """
        Initialize the classification decoder.

        Parameters
        ----------
        embedding_dim : int
            Dimensionality of the input embedding.
        n_cls : int
            Number of target classes.
        dropout : float, optional
            Dropout probability used in the classifier.
        """
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, n_cls)
        )

    def forward(self, x: Tensor):
        """
        Compute class logits from embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input embedding tensor of shape (batch_size, embedding_dim).

        Returns
        -------
        torch.Tensor
            Classification logits of shape (batch_size, n_cls).
        """
        o = self.fc(x)
        return o


class BatchDecoder(nn.Module):
    """
    Discriminator head for batch prediction with gradient reversal.

    This module predicts batch labels from embedding representations and
    applies a gradient reversal layer (GRL) during backpropagation to
    encourage batch-invariant features.
    """
    def __init__(
        self,
        embedding_dim: int,
        n_batch: int,
        dropout: float = 0.1
    ):
        """
        Initialize the batch discriminator.

        Parameters
        ----------
        embedding_dim : int
            Dimensionality of the input embedding.
        n_batch : int
            Number of batch classes.
        dropout : float, optional
            Dropout probability used in the discriminator MLP.
        """
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, n_batch)
        )

    def forward(self, x: Tensor, alpha=1.0):
        """
        Forward pass with gradient reversal.

        Parameters
        ----------
        x : torch.Tensor
            Input embedding tensor of shape (batch_size, embedding_dim).
        alpha : float, optional
            Gradient reversal scaling factor.

        Returns
        -------
        torch.Tensor
            Batch classification logits of shape (batch_size, n_batch).
        """
        x_rev = grad_reverse(x, alpha)
        o = self.fc(x_rev)
        return o


class GeneNameEncoder(nn.Module):
    """
    Encode gene identity tokens into normalized embedding representations.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        """
        Initialize the gene name embedding encoder.

        Parameters
        ----------
        num_embeddings : int
            Vocabulary size (number of unique gene tokens).
        embedding_dim : int
            Dimensionality of the embedding vectors.
        padding_idx : int, optional
            Index treated as padding; its embedding will not be updated.
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode gene ID sequences into embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Gene ID tensor of shape (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            Normalized embedding tensor of shape
            (batch_size, seq_len, embedding_dim).
        """
        x = self.embedding(x)
        x = self.enc_norm(x)
        return x

    
class GeneExpValueEncoder(nn.Module):
    """
    Encode continuous expression values into embedding vectors.
    """

    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_value: int = 512):
        """
        Initialize the expression value encoder.

        Parameters
        ----------
        embedding_dim : int
            Output embedding dimension.
        dropout : float, optional
            Dropout probability applied to the output embeddings.
        max_value : int, optional
            Upper bound used to clip input values for numerical stability.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, embedding_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode expression values into embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Expression value tensor of shape (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            Embedded expression tensor of shape
            (batch_size, seq_len, embedding_dim).
        """
        x = x.unsqueeze(-1)
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class GeneChrEncoder(nn.Module):
    """
    Encode chromosome indices into normalized embedding representations.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        """
        Initialize the chromosome embedding encoder.

        Parameters
        ----------
        num_embeddings : int
            Total number of chromosome indices.
        embedding_dim : int
            Dimensionality of the embedding vectors.
        padding_idx : int, optional
            Index treated as padding; its embedding will not be updated.
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode chromosome indices into embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Chromosome index tensor of shape (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            Normalized embedding tensor of shape
            (batch_size, seq_len, embedding_dim).
        """
        x = self.embedding(x)
        x = self.enc_norm(x)
        return x


class GenePosEncoder(nn.Module):
    """
    Generate sinusoidal positional encodings based on gene positions.
    """
    def __init__(self, div_term_scale=1000):
        super(GenePosEncoder, self).__init__()
        self.div_term_scale = div_term_scale

    def forward(self, x, positions):
        """
        Compute positional encodings for input embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, seq_len, emb_size) or
            (batch_size, seq_len, emb_size). If a singleton dimension is
            present, it will be squeezed.
        positions : torch.Tensor
            Position tensor of shape (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            Positional encoding tensor of shape
            (batch_size, seq_len, emb_size).
        """
        x = x.squeeze(1)

        batchsize, seq_len, embsize = x.size()

        position_encoding = torch.zeros(batchsize, seq_len, embsize, device=x.device)

        div_term = torch.pow(
            10000,
            torch.arange(0., embsize, 2, device=x.device) / embsize
        ) / self.div_term_scale

        for i in range(embsize // 2):
            position_encoding[:, :, 2 * i] = torch.sin(positions * div_term[i])
            position_encoding[:, :, 2 * i + 1] = torch.cos(positions * div_term[i])

        return position_encoding
