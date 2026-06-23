from typing import Any, Callable, List, Optional, Union
import os
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from deepspeed.moe.layer import MoE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
try:
    from flash_attn.flash_attention import FlashMHA

    flash_attn_available = True
except ImportError:
    import warnings

    warnings.warn("flash_attn is not installed")
    flash_attn_available = False

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
torch.cuda.empty_cache()


class GeneNameEncoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = self.enc_norm(x)
        return x


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
        x = self.embedding(x)
        x = self.enc_norm(x)
        return x


class GenePosEncoder(nn.Module):
    def __init__(self, div_term_scale=1000):
        super().__init__()
        self.div_term_scale = div_term_scale

    def forward(self, x, positions):
        x = x.squeeze(1)
        batchsize, seq_len, embsize = x.size()
        position_encoding = torch.zeros(batchsize, seq_len, embsize, device=x.device)
        div_term = torch.pow(10000, torch.arange(0.0, embsize, 2, device=x.device) / embsize) / self.div_term_scale

        for i in range(embsize // 2):
            position_encoding[:, :, 2 * i] = torch.sin(positions * div_term[i])
            position_encoding[:, :, 2 * i + 1] = torch.cos(positions * div_term[i])

        return position_encoding


class PertEncoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        x = self.embedding(x)
        x = self.enc_norm(x)
        return x


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


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
        pred_value = self.fc(x).squeeze(-1)
        return dict(pred=pred_value)


class ClsDecoder(nn.Module):
    def __init__(self, embedding_dim: int, n_cls: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, n_cls),
        )

    def forward(self, x: Tensor):
        return self.fc(x)


class BatchDecoder(nn.Module):
    def __init__(self, embedding_dim: int, n_batch: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, n_batch),
        )

    def forward(self, x: Tensor, alpha=1.0):
        x_rev = grad_reverse(x, alpha)
        o = self.fc(x_rev)
        return o


class Cls2ExpValueDecoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x: Tensor):
        y = self.decoder(x)
        return y


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
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


class MoETransformerEncoderLayer(nn.Module):
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 128, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None, moe_experts: int = 6, epsize: int = 1,top_k: int = 2) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first, **factory_kwargs)
        self.moe_ffn = MoE(
            hidden_size=d_model,
            expert=ExpertModel(d_model, dim_feedforward, dropout),
            num_experts=moe_experts,
            ep_size = epsize,
            k=top_k
        )

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        self.activation = activation

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None, is_causal: bool = False) -> Tensor:
        src_key_padding_mask = F._canonical_mask(mask=src_key_padding_mask, mask_name="src_key_padding_mask",
                                                 other_type=F._none_or_dtype(src_mask), other_name="src_mask",
                                                 target_type=src.dtype)
        src_mask = F._canonical_mask(mask=src_mask, mask_name="src_mask", other_type=None, other_name="",
                                     target_type=src.dtype, check_other=False)

        x = src
        if self.norm_first:
            pass
        else:
            out, attn_weights = self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            x = self.norm1(x + out)
            x_cls = x[:, 0, :]
            y, l_aux, exp_count,custom_output,combine_weights,original_logits = self._ff_block(x)
            x = self.norm2(y + x)

            batchsize, seqlen, embsize1 = x.shape
            e, bsl, embsize2 = custom_output.shape

            assert batchsize * seqlen == bsl, "Unexpected custom output shape."
            assert embsize1 == embsize2, "Embedding dimension mismatch."

            custom_output = custom_output.reshape(e, batchsize, seqlen, embsize2)
            custom_output = custom_output[:, :, 0, :]

            custom_output = self.norm2(custom_output + x_cls)
            combine_weights = combine_weights.reshape(batchsize, seqlen,e,-1)
            gate_cls = combine_weights[:, 0, :, :]
            gate_cls = gate_cls.sum(dim=-1)
            original_logits = original_logits.reshape(batchsize, seqlen, e)
            gate_logits_cls = original_logits[:,0,:]

        return x, l_aux, exp_count, custom_output,gate_cls,gate_logits_cls, attn_weights
    

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], 
                  is_causal: bool = False) -> Tensor:
        attn_output, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal
            )
        return self.dropout1(attn_output), attn_weights

    def _ff_block(self, x: Tensor) -> Tensor:
        moe_out,l_aux, exp_count,custom_output,gate_weight,original_logits = self.moe_ffn(x)
        return self.dropout2(moe_out),l_aux, exp_count, custom_output,gate_weight,original_logits
    




class MoETransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, layer_norm_eps: float = 1e-5,
                 batch_first: bool = True, norm_first: bool = False, bias: bool = True, device=None, dtype=None,
                 moe_experts: int = 6, epsize: int = 1, top_k: int = 2,
                 moe_layers: List[int] = [3, 5, 7, 9, 11]) -> None:
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i in moe_layers:
                self.layers.append(
                    MoETransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                               layer_norm_eps, batch_first, norm_first, bias, device, dtype,
                                               moe_experts, epsize, top_k)
                )
            else:
                self.layers.append(
                    nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                               activation=activation, layer_norm_eps=layer_norm_eps,
                                               batch_first=batch_first, norm_first=norm_first, bias=bias,
                                               device=device, dtype=dtype)
                )
    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None, is_causal: bool = False) -> Tensor:
        x = src
        aux_losses = []
        expert_counts = []
        gate_weight_cls_all_layers = []
        gate_logits_cls_all_layers = []
        attn_weights_all_layers = []
        for layer in self.layers:
            if isinstance(layer, MoETransformerEncoderLayer):
                x, l_aux, exp_count,custom_output,gate_weight_cls,gate_logits_cls, attn_weights = layer(x, src_mask, src_key_padding_mask, is_causal)
                aux_losses.append(l_aux)
                expert_counts.append(exp_count)
                gate_weight_cls_all_layers.append(gate_weight_cls)
                gate_logits_cls_all_layers.append(gate_logits_cls)
                attn_weights_all_layers.append(attn_weights)
            else:
                x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)
        
        return x, aux_losses, expert_counts,custom_output,gate_weight_cls, gate_logits_cls, attn_weights, gate_weight_cls_all_layers,gate_logits_cls_all_layers, attn_weights_all_layers


class TransformerModel(nn.Module):
    def __init__(
        self, 
        ntokens: int,
        num_layers: int, 
        d_model: int, 
        nhead: int,  
        seq_len: int,
        n_cls: int,
        n_batch: int,
        pad_token: str = '<pad>',
        vocab: Any = None,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, 
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True, 
        norm_first: bool = False, 
        bias: bool = True, 
        device=None, 
        dtype=None,
        moe = True,
        GRL: bool = False,
        mlm: bool = False,
        CL: bool = False,
        cls: bool = False,
        guide_expert: bool =True,
        pert: bool = True,
        dim_feedforward: int = 1024,
        aux_lambda: float = 0.8,
        alpha: float = 1,
        beta: float = 1,
        moe_experts: int = 6, 
        epsize: int = 1, # 
        top_k: int = 1,
        return_emb_when_inf: Union[None, str] = "cell",
        custom_gene_path = None,
        moe_layers: List[int] = [3, 5, 7, 9, 11]) -> None:

        super().__init__()

        self.GRL = GRL
        self.mlm = mlm
        self.cls = cls
        self.CL = CL
        self.guide_expert = guide_expert
        self.pert = pert
        self.initial_lambda = aux_lambda
        self.aux_lambda = aux_lambda
        self.alpha = alpha
        self.beta = beta
        self.return_emb_when_inf = return_emb_when_inf
        self.custom_gene_path = custom_gene_path
    
        self.encoder1 = GeneNameEncoder(num_embeddings = ntokens+1, embedding_dim = d_model, padding_idx = vocab[pad_token])
        self.encoder2 = GeneExpValueEncoder(embedding_dim = d_model)
        self.encoder3 = GeneChrEncoder(num_embeddings = 27, embedding_dim = d_model, padding_idx = 26)
        self.encoder4 = GenePosEncoder()
        self.encoder5 = PertEncoder(num_embeddings=3, embedding_dim =d_model, padding_idx=2)


        self.transformer_enc = MoETransformerEncoder(num_layers=num_layers, d_model = d_model, nhead=nhead,
                                                     dim_feedforward=dim_feedforward,
                                                     moe_experts = moe_experts,epsize = epsize, top_k=top_k,moe_layers = moe_layers)

        if self.GRL:
            self.decoder4 = BatchDecoder(embedding_dim = d_model, n_batch = n_batch)

        if self.mlm:
            self.decoder1 = GeneExpValueDecoder(embedding_dim = d_model)
            self.decoder3 = Cls2ExpValueDecoder(embedding_dim = d_model)

        if self.cls:
            self.decoder2 = ClsDecoder(embedding_dim = d_model, n_cls = n_cls)

        if self.pert:
            self.decoder1 = GeneExpValueDecoder(embedding_dim = d_model)
            self.decoder3 = Cls2ExpValueDecoder(embedding_dim = d_model)    

    
    def masked_mse_loss(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float()
        loss = F.mse_loss(input * mask, target * mask, reduction="sum")
        return loss / mask.sum()

    def cls_loss(self, output_cls: torch.Tensor, target_cls: torch.Tensor)-> torch.Tensor:
        criterion_cls = nn.CrossEntropyLoss()
        loss = criterion_cls(output_cls, target_cls)
        return loss
    
    def discriminator_loss(self, output_dis: torch.Tensor, target_dis: torch.Tensor)-> torch.Tensor:
        criterion_dis = nn.CrossEntropyLoss()
        loss = criterion_dis(output_dis, target_dis)
        preds = torch.argmax(output_dis, dim=1)
        correct = (preds == target_dis).sum().item()
        total = target_dis.size(0)
        acc = correct / total
        return loss, acc
    

    
    def compute_accuracy(self,logits: torch.Tensor, targets: torch.Tensor) -> float:
        preds = torch.argmax(logits, dim=1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)
        return correct / total

    def compute_metrics(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        labels = targets.cpu().numpy()
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='macro', zero_division=0)
        recall = recall_score(labels, preds, average='macro', zero_division=0)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)

        return accuracy, precision, recall, f1
    


    def contrastive_loss_nt_xent(self, cls1: torch.Tensor,
                                cls2: torch.Tensor,
                                temperature: float,
                                bidirectional: bool = True) -> torch.Tensor:
        """
        SimCLR-style NT-Xent (InfoNCE) loss for two views of CLS embeddings.

        Args:
            cls1, cls2: [B, hidden_dim] two sets of embeddings (two views of the same batch)
            temperature: float, softmax temperature
            bidirectional: bool, whether to compute both cls1->cls2 and cls2->cls1

        Returns:
            loss: scalar contrastive loss
        """
        B = cls1.shape[0]

        # Normalize embeddings
        cls1_norm = F.normalize(cls1, dim=-1)
        cls2_norm = F.normalize(cls2, dim=-1)

        # Cosine similarity matrix [B, B]
        logits = torch.matmul(cls1_norm, cls2_norm.T) / temperature  # (B,B)

        # Labels: each sample i in cls1 matches cls2[i]
        labels = torch.arange(B, device=cls1.device)

        # cls1 -> cls2
        loss12 = F.cross_entropy(logits, labels)

        if bidirectional:
            # cls2 -> cls1
            logits_T = torch.matmul(cls2_norm, cls1_norm.T) / temperature
            loss21 = F.cross_entropy(logits_T, labels)
            loss = 0.5 * (loss12 + loss21)
        else:
            loss = loss12

        return loss
    def gate_loss(self, gate_logits, batch_id):
        log_p_A = torch.logsumexp(gate_logits[:, 0:3], dim=1)  # [B]
        log_p_B = torch.logsumexp(gate_logits[:, 3:6], dim=1)  # [B]

        group_logits = torch.stack([log_p_A, log_p_B], dim=1)  # [B,2]
        loss = F.cross_entropy(group_logits, batch_id)
        return loss


    def gate_metrics(self, gate_logits, batch_id):
        log_p_A = torch.logsumexp(gate_logits[:, 0:3], dim=1)  # [B]
        log_p_B = torch.logsumexp(gate_logits[:, 3:6], dim=1)  # [B]
        group_logits = torch.stack([log_p_A, log_p_B], dim=1)  # [B,2]

        pred = group_logits.argmax(dim=1)
        target = batch_id.long()
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

        accuracy = accuracy_score(target,pred)
        precision = precision_score(target, pred, average='macro', zero_division=0)
        recall = recall_score(target, pred, average='macro', zero_division=0)
        f1 = f1_score(target, pred, average='macro', zero_division=0)

        return accuracy, precision, recall, f1




    def compute_pert_metrics(
        self,
        ctrl_exp,       # [batch, seq_len]
        target_exp,     # [batch, seq_len]
        pred_exp,       # [batch, seq_len]
    ):
        metrics = {}

        exp_mask_bool_all = torch.ones_like(pred_exp, dtype=torch.bool)
        exp_mask_bool_all[:, 0] = False

        mse_all = self.masked_mse_loss(pred_exp, target_exp, exp_mask_bool_all)
        metrics["mse_all"] = mse_all

        pearson_all = self.batchwise_pearson(
            pred_exp[:, 1:],
            target_exp[:, 1:]
        )
        metrics["pearson_all"] = pearson_all

        delta_pred = pred_exp - ctrl_exp
        delta_target = target_exp - ctrl_exp
        metrics["delta_pearson_all"] = self.batchwise_pearson(
            delta_pred[:, 1:],
            delta_target[:, 1:]
        )

        return metrics
    
    def batchwise_pearson(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)

        x_centered = x - x_mean
        y_centered = y - y_mean

        numerator = (x_centered * y_centered).sum(dim=1)
        denominator = (
            x_centered.pow(2).sum(dim=1).sqrt() * y_centered.pow(2).sum(dim=1).sqrt()
        )

        pearson = numerator / (denominator + 1e-8)
        valid_mask = ~torch.isnan(pearson)

        if valid_mask.sum() == 0:
            return torch.tensor(float("nan"), device=x.device)

        return pearson[valid_mask].mean()





    def forward(self, return_cell_emb=False, inf=False, **inputs):
        if inf==False:
            gene_id = inputs["gene_id"]
            pos = inputs["pos"]
            chr = inputs["chr"]
            src_key_padding_mask = inputs["src_key_padding_mask"]
            ctrl_exp = inputs['ctrl_exp']
            target_exp = inputs["target_exp"]
            pert_mtx = inputs['pert_mtx']


            gene_ids_enc = self.encoder1(gene_id)

            if self.mlm:
                masked_exp_1 = inputs["masked_exp_1"]
                exp_mask_bool_1 = inputs["exp_mask_bool_1"]
                masked_exp_1 = masked_exp_1.squeeze(1)  
                exp_mask_bool_1 = exp_mask_bool_1.squeeze(1) 
                exp_value_1_enc = self.encoder2(masked_exp_1)


            if self.CL:
                masked_exp_2 = inputs["masked_exp_2"]
                exp_mask_bool_2 = inputs["exp_mask_bool_2"]
                masked_exp_2 = masked_exp_2.squeeze(1)  
                exp_mask_bool_2 = exp_mask_bool_2.squeeze(1) 
                exp_value_2_enc = self.encoder2(masked_exp_2)

            if self.pert:
                ctrl_exp = inputs["ctrl_exp"]
                ctrl_exp_enc = self.encoder2(ctrl_exp) 



            chr_enc = self.encoder3(chr)
            pos_enc = self.encoder4(ctrl_exp_enc, pos)
            pert_mtx_enc = self.encoder5(pert_mtx)



            chr_enc = chr_enc.to(dtype=next(self.parameters()).dtype)
            pos_enc = pos_enc.to(dtype=next(self.parameters()).dtype)

        

            total_enc_1 = gene_ids_enc + ctrl_exp_enc + chr_enc + pos_enc + pert_mtx_enc

            if self.CL:
                total_enc_2 = gene_ids_enc + exp_value_2_enc + chr_enc + pos_enc 

    


            # output from transformer
            (
                output1_view1,
                aux_losses_view1,
                expert_counts_view1,
                custom_outputs_view1,
                gate_weight_cls_view1,
                gate_logits_cls_view1,
                attn_weights_view1,
                gate_weight_cls_all_layers_view1,
                gate_logits_cls_all_layers_view1,
                attn_weights_all_layers_view1,
            ) = self.transformer_enc(
                total_enc_1,
                src_key_padding_mask=src_key_padding_mask
            ) # (batchsz,seqlen,d_model)
            
            
            if self.CL:
                (
                    output1_view2,
                    aux_losses_view2,
                    expert_counts_view2,
                    custom_outputs_view2,
                    gate_weight_cls_view2,
                    gate_logits_cls_view2,
                    attn_weights_view2,
                    gate_weight_cls_all_layers_view2,
                    gate_logits_cls_all_layers_view2,
                    attn_weights_all_layers_view2,
                ) = self.transformer_enc(
                    total_enc_2,
                    src_key_padding_mask=src_key_padding_mask
                ) # (batchsz,seqlen,d_model)
                
            cell_emb_view1 = output1_view1[:, 0, :]
            if self.CL:
                cell_emb_view2 = output1_view2[:, 0, :]
            
            l_aux_total = sum(aux_losses_view1) / len(aux_losses_view1)

            if self.mlm:
                output2_view1 = self.decoder1(output1_view1)
                mse_loss_view1 = self.masked_mse_loss(output2_view1["pred"], target_exp, exp_mask_bool_1)
                output3_view1 = self.decoder3(cell_emb_view1)
                output1_t_view1 = output1_view1.transpose(1, 2)
                pred_exp_bycls_view1 = torch.einsum('bd, bdn -> bn', output3_view1, output1_t_view1)
                mse_loss_bycls_view1 = self.masked_mse_loss(pred_exp_bycls_view1, target_exp, exp_mask_bool_1)
                total_loss = mse_loss_view1 + mse_loss_bycls_view1

            if self.pert:
                output2_view1 = self.decoder1(output1_view1)
                metrics = self.compute_pert_metrics(ctrl_exp, target_exp, output2_view1["pred"])
                mse_loss = metrics['mse_all']
                pearson = metrics['pearson_all']
                delta_pearson = metrics['delta_pearson_all']
                
                output3_view1 = self.decoder3(cell_emb_view1)
                output1_t_view1 = output1_view1.transpose(1, 2)
                pred_exp_bycls_view1 = torch.einsum('bd, bdn -> bn', output3_view1, output1_t_view1)
                metrics_bycls = self.compute_pert_metrics(ctrl_exp, target_exp, pred_exp_bycls_view1)
                mse_loss_bycls = metrics_bycls['mse_all']
                total_loss = mse_loss + mse_loss_bycls

            if isinstance(l_aux_total, torch.Tensor):
                l_aux_total = l_aux_total.item()
            else:
                l_aux_total = float(l_aux_total)


            if self.GRL:
                discriminator_pred = self.decoder4(cell_emb_view1)
                discriminator_loss, discriminator_acc = self.discriminator_loss(discriminator_pred, discriminator_target)
                total_loss += self.beta * discriminator_loss

            if self.CL:
                contrastive_loss = self.contrastive_loss_nt_xent(cell_emb_view1, cell_emb_view2, temperature=0.1)
                total_loss += self.alpha * contrastive_loss

            
            if self.cls:
                cls_pred = self.decoder2(cell_emb)
                cls_loss = self.cls_loss(cls_pred, cls_target)
                total_loss = cls_loss + (l_aux_total - 1) ** 2
                accuracy, precision, recall, f1 = self.compute_metrics(cls_pred, cls_target)
                if self.guide_expert:
                    gate_loss = self.gate_loss(gate_logits_cls,batch_id)
                    total_loss += gate_loss * 1
                    gate_accuracy, gate_precision, gate_recall, gate_f1 = self.gate_metrics(gate_logits_cls,batch_id) 

            total_loss += self.aux_lambda * l_aux_total


            if isinstance(l_aux_total, torch.Tensor):
                l_aux_total = l_aux_total.item()
            else:
                l_aux_total = float(l_aux_total)


            if self.cls == True:
                return {"cls_loss": cls_loss, "total_loss": total_loss, "accuracy":accuracy, "precision":precision,
                        "recall":recall,"f1":f1,
                        "gate_accuracy":gate_accuracy, "gate_precision":gate_precision,
                        "gate_recall":gate_recall,"gate_f1":gate_f1,
                        "l_aux": l_aux_total.item() if hasattr(l_aux_total, "item") else l_aux_total,
                        "aux_lambda": self.aux_lambda.item() if hasattr(self.aux_lambda, "item") else self.aux_lambda, 
                        "logits": cls_pred,"cell_emb": cell_emb.detach().cpu() , "labels": cls_target.detach().cpu()}
            elif self.mlm and self.CL and self.GRL:
                # if self.mlm and self.CL and self.GRL:
                return {"total_loss": total_loss, "mse_loss": mse_loss_view1, "mse_loss_bycls": mse_loss_bycls_view1, 
                        "contrastive_loss": contrastive_loss, "discriminator_loss": discriminator_loss,
                        "discriminator_acc":discriminator_acc,
                        "l_aux": l_aux_total.item() if hasattr(l_aux_total, "item") else l_aux_total,
                        "aux_lambda": self.aux_lambda.item() if hasattr(self.aux_lambda, "item") else self.aux_lambda}
            elif self.mlm and self.CL:
                return {"total_loss": total_loss, "mse_loss": mse_loss_view1, "mse_loss_bycls": mse_loss_bycls_view1, 
                        "contrastive_loss": contrastive_loss,
                        "l_aux": l_aux_total.item() if hasattr(l_aux_total, "item") else l_aux_total,
                        "aux_lambda": self.aux_lambda.item() if hasattr(self.aux_lambda, "item") else self.aux_lambda}
            elif self.mlm and self.GRL:
                return {"total_loss": total_loss, "mse_loss": mse_loss_view1, "mse_loss_bycls": mse_loss_bycls_view1, 
                        "discriminator_loss": discriminator_loss,
                        "discriminator_acc":discriminator_acc,
                        "l_aux": l_aux_total.item() if hasattr(l_aux_total, "item") else l_aux_total,
                        "aux_lambda": self.aux_lambda.item() if hasattr(self.aux_lambda, "item") else self.aux_lambda}
            elif self.mlm:
                return {"total_loss": total_loss, "mse_loss": mse_loss_view1, "mse_loss_bycls": mse_loss_bycls_view1, 
                        "l_aux": l_aux_total.item() if hasattr(l_aux_total, "item") else l_aux_total,
                        "aux_lambda": self.aux_lambda.item() if hasattr(self.aux_lambda, "item") else self.aux_lambda}

            elif self.pert:
                return {"total_loss": total_loss, "mse_loss": mse_loss, "mse_loss_bycls": mse_loss_bycls, 
                        "pearson":pearson,
                        "delta_pearson": delta_pearson,
                        "l_aux": l_aux_total.item() if hasattr(l_aux_total, "item") else l_aux_total,
                        "aux_lambda": self.aux_lambda.item() if hasattr(self.aux_lambda, "item") else self.aux_lambda}
            


        if inf==True:
            gene_id = inputs["gene_id"]
            pos = inputs["pos"]
            chr = inputs["chr"]
            src_key_padding_mask = inputs["src_key_padding_mask"]
            ctrl_exp = inputs['ctrl_exp']
            target_exp = inputs["target_exp"]
            pert_mtx = inputs['pert_mtx']


            if self.mlm:
                pass



            gene_ids_enc = self.encoder1(gene_id)
            ctrl_exp_enc = self.encoder2(ctrl_exp) 
            chr_enc = self.encoder3(chr)
            pos_enc = self.encoder4(ctrl_exp_enc, pos)
            pert_mtx_enc = self.encoder5(pert_mtx)


            chr_enc = chr_enc.to(dtype=next(self.parameters()).dtype)
            pos_enc = pos_enc.to(dtype=next(self.parameters()).dtype)

            total_enc = gene_ids_enc + ctrl_exp_enc + chr_enc + pos_enc + pert_mtx_enc

    

            # output from transformer
            (
                output1,
                aux_losses,
                expert_counts,
                custom_outputs,
                gate_weight_cls,
                gate_logits_cls,
                attn_weights,
                gate_weight_cls_all_layers,
                gate_logits_cls_all_layers,
                attn_weights_all_layers,
            ) = self.transformer_enc(
                total_enc,
                src_key_padding_mask=src_key_padding_mask
            )

            cell_emb = output1[:, 0, :]

            if self.return_emb_when_inf == 'cell':
                rt_emb = cell_emb
            elif self.return_emb_when_inf == 'all':
                rt_emb = output1
            elif self.return_emb_when_inf == 'custom':
                import json
                with open(self.custom_gene_path, "r") as f:
                    gene_dict = json.load(f)
                    gene_custom_indices = [idx + 1 for idx in gene_dict.values()]
                    cls_index = 0
                    all_indices = [cls_index] + gene_custom_indices
                    idx_tensor = torch.tensor(all_indices, device=output1.device)
                    rt_emb = torch.index_select(output1, dim=1, index=idx_tensor)

            else:
                rt_emb = None

            cell_emb_n_expert = custom_outputs
            l_aux_total = sum(aux_losses) / len(aux_losses)

            if self.mlm:
                pass

            if self.cls:
                cls_pred = self.decoder2(cell_emb)

            if self.pert:
                output2 = self.decoder1(output1)
                pred_exp_value = output2["pred"][:, 1:]
            
            if isinstance(l_aux_total, torch.Tensor):
                l_aux_total = l_aux_total.item()
            else:
                l_aux_total = float(l_aux_total)
            if self.return_emb_when_inf == 'all':
                rt_emb = rt_emb.mean(dim=0) # [seq_len, emb_size]
            elif self.return_emb_when_inf == 'custom':
                rt_emb = rt_emb.half() # [batch_size, custom_gene_num + 1, emb_size]
            elif self.return_emb_when_inf == 'cell':
                rt_emb = rt_emb  # [batch_size, emb_size]
            else:
                rt_emb = None



            if self.cls == True:
                return { "l_aux": l_aux_total.item() if hasattr(l_aux_total, "item") else l_aux_total,
                        "aux_lambda": self.aux_lambda.item() if hasattr(self.aux_lambda, "item") else self.aux_lambda, 
                        "logits": cls_pred,"cell_emb": cell_emb.detach().cpu(),"expert_counts" : expert_counts,
                        "gate_weight_cls":gate_weight_cls,
                        "cell_emb_n_expert":cell_emb_n_expert.detach().cpu()}
            else:
                return {
                        "l_aux": l_aux_total.item() if hasattr(l_aux_total, "item") else l_aux_total,
                        # "pred_exp_bycls":pred_exp_bycls.detach().cpu(),
                        # "pred_exp_bygene":output2["pred"].detach().cpu(),
                        "pred_exp_value": pred_exp_value,
                        "aux_lambda": self.aux_lambda.item() if hasattr(self.aux_lambda, "item") else self.aux_lambda, 
                        # "attn_weights_celltype":attn_weights_celltype.detach().cpu(),
                        # "rt_emb": rt_emb.detach().cpu(),
                        # "gate_weight_cls":gate_weight_cls
                        # "cell_emb_n_expert":cell_emb_n_expert.detach().cpu()
                        }

    
    
