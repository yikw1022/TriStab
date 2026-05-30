"""
娑堣瀺瀹為獙 A3: Triadic -> Pair

鐩告瘮瀹屾暣妯″瀷 (hyperppn_lite_oetr_r1)锛?  - 灏?OETRTriadicLayerR1锛堜笁浣撲氦浜掞級鏇挎崲涓?PairwiseLayer锛堟垚瀵逛氦浜掞級
  - 涓嶆瀯閫犻偦灞呭 (j_p, j_q)锛屼粎淇濈暀涓績浣嶇偣涓庡崟閭诲眳鐨勬垚瀵规秷鎭紶閫?  - 鍏朵綑缁撴瀯瀹屽叏涓嶅彉锛堝弻鍒嗘敮 + 铻嶅悎灞傦級
  - 鏍稿績楠岃瘉"楂橀樁涓変綋"鏄惁浼樹簬鏅€氬眬閮ㄦ秷鎭紶閫?
娉ㄥ唽涓?'ablation_triadic2pair'
"""
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from tristab.models import register_model
from tristab.models.stability.basemodel import BaseModel
from tristab.models.stability.protein_mpnn import ProteinMPNNConfig
from tristab.models.stability.modules.esm2 import ESM2
from tristab.models.stability.org_transfer_model import get_protein_mpnn
from tristab import utils
log = utils.get_logger(__name__)


class PairwiseLayer(nn.Module):
    """銆愭秷铻嶆浛浠ｅ眰銆戜腑蹇冧綅鐐?脳 鍗曢偦灞呯殑鎴愬浜や簰灞傘€?
    涓?OETRTriadicLayerR1 鐨勫尯鍒細
      - 涓嶆瀯閫?T脳T 鐨勯偦灞呭鐭╅樀锛屼粎寤烘ā (mut_token, ctx_j, e_j) 鐨勬垚瀵瑰叧绯?      - 杈撳叆缁村害锛?d锛堣€岄潪涓変綋鐨?5d+1锛?      - 鐢ㄥ彲瀛︿範娉ㄦ剰鍔涘仛閭诲眳鑱氬悎
    鎺ュ彛涓?OETRTriadicLayerR1 瀹屽叏瀵归綈銆?    """

    def __init__(self, d_model, d_edge, T=4, triadic_dropout=0.3):
        super().__init__()
        self.T = T
        self.d = d_model
        self.edge_proj = nn.Linear(d_edge, d_model)
        # 鎴愬娑堟伅鍑芥暟: [mut_token, ctx_j, e_j] => 3d
        self.msg_fn = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(triadic_dropout),
            nn.Linear(d_model * 2, d_model),
        )
        # Attention score for each neighbor.
        self.attn_fn = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(triadic_dropout),
            nn.Linear(d_model, 1),
        )
        # 闂ㄦ帶娈嬪樊
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)

    def forward(self, mut_token, ctx, edge_feat, neighbors, nb_mask):
        T = min(self.T, neighbors.shape[1])
        M, d = mut_token.shape
        nb_idx = neighbors[:, :T]
        mask = nb_mask[:, :T]
        nb_ctx = ctx[nb_idx]                              # [M, T, d]
        e_proj = self.edge_proj(edge_feat[:, :T])          # [M, T, d]
        h = mut_token.unsqueeze(1).expand(-1, T, -1)       # [M, T, d]
        pair_input = torch.cat([h, nb_ctx, e_proj], dim=(-1))  # [M, T, 3d]
        # 鎴愬娑堟伅
        msg_j = self.msg_fn(pair_input)                    # [M, T, d]
        # 娉ㄦ剰鍔涙潈閲?        attn_logits = self.attn_fn(pair_input).squeeze(-1)  # [M, T]
        attn_logits = attn_logits.masked_fill(~mask, -10000.0)
        alpha = torch.softmax(attn_logits, dim=(-1))        # [M, T]
        # 鍔犳潈鑱氬悎
        msg = (msg_j * alpha.unsqueeze(-1)).sum(dim=1)      # [M, d]
        # 闂ㄦ帶娈嬪樊 + LayerNorm
        gate = self.gate(torch.cat([mut_token, msg], dim=(-1)))
        mut_token = self.norm(mut_token + gate * msg)
        return mut_token


@dataclass
class ConfigAblationTriadic2Pair:
    encoder_mpnn = field(default=(ProteinMPNNConfig()))
    encoder_mpnn: ProteinMPNNConfig
    esm_name = "esm2_t33_650M_UR50D"
    esm_name: str
    mpnn_name = "ProteinMPNN"
    mpnn_name: str
    dropout = 0.1
    dropout: float
    esm_tune = False
    esm_tune: bool
    mpnn_tune = True
    mpnn_tune: bool
    d_model = 256
    d_model: int
    d_edge = 128
    d_edge: int
    T = 4
    T: int
    n_triadic = 2
    n_triadic: int
    triadic_dropout = 0.3
    triadic_dropout: float


@register_model("ablation_triadic2pair")
class AblationTriadic2Pair(BaseModel):
    _default_cfg = ConfigAblationTriadic2Pair()

    def __init__(self, cfg):
        super().__init__(cfg)
        self.esm_decoder = ESM2.from_pretrained(args=(self.cfg), name=(self.cfg.esm_name))
        self.padding_idx = self.esm_decoder.padding_idx
        self.mpnn_encoder = get_protein_mpnn(tune=(cfg.mpnn_tune))
        self.use_input_decoding_order = cfg.encoder_mpnn.use_input_decoding_order
        encoder_layer = nn.TransformerEncoderLayer(d_model=1792,
          nhead=8,
          dropout=0.1,
          dim_feedforward=3584,
          batch_first=True)
        self.transformer_encoder_protein = nn.TransformerEncoder(encoder_layer, num_layers=2)
        d = getattr(self.cfg, "d_model", 256)
        d_edge = getattr(self.cfg, "d_edge", 128)
        T = getattr(self.cfg, "T", 4)
        n_triadic = getattr(self.cfg, "n_triadic", 2)
        triadic_dropout = getattr(self.cfg, "triadic_dropout", 0.3)
        # Stream A: 浣嶇偣鍋忓ソ MLP锛堜繚鐣欎笉鍙橈級
        self.fitness_mlp = nn.Sequential(nn.Linear(1792, 1024), nn.GELU(), nn.Dropout(0.2), nn.Linear(1024, 21))
        # Stream B: 缁撴瀯鎵板姩鍒嗘敮
        self.site_proj = nn.Linear(1792, d)
        self.logits_proj = nn.Linear(20, d)
        self.wt_emb = nn.Embedding(21, d)
        self.mt_emb = nn.Embedding(21, d)
        self.mut_fuse = nn.Sequential(nn.Linear(d * 4, d * 2), nn.GELU(), nn.Linear(d * 2, d))
        self.ctx_proj = nn.Linear(1792, d)
        # 銆愭秷铻嶃€戠敤 PairwiseLayer 鏇挎崲 OETRTriadicLayerR1
        self.triadic_layers = nn.ModuleList([PairwiseLayer(d, d_edge, T, triadic_dropout=triadic_dropout) for _ in range(n_triadic)])
        self.perturb_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d // 2, 1))
        # Fusion锛堜繚鐣欎笉鍙橈級
        self.ddg_out = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Dropout(0.1), nn.Linear(16, 1))

    def forward(self, batch, **kwargs):
        with torch.set_grad_enabled(self.cfg.mpnn_tune):
            mpnn_features, E_idx, h_E, edge_valid_mask = self.forward_mpnn(batch, return_graph=True)
        with torch.set_grad_enabled(self.cfg.esm_tune):
            esm_out = self.esm_decoder(tokens=(batch["tokens"]), encoder_out=None)
            esm_features = esm_out["representations"][-1][:, 1:-1]
            esm_logits = esm_out["logits"][:, 1:-1]
        protein_feature = torch.cat([esm_features, mpnn_features], dim=(-1))
        protein_feature = self.transformer_encoder_protein(protein_feature)
        pf = protein_feature[0]
        el = esm_logits[0]
        e_idx = E_idx[0] if E_idx.dim() == 3 else E_idx
        e_mask = edge_valid_mask[0] if edge_valid_mask.dim() == 3 else edge_valid_mask
        h_e = h_E[0] if h_E.dim() == 4 else h_E
        esm_logits_20 = F.log_softmax(el, dim=(-1))[:, 4:24]
        mut_ids = batch["mut_ids"]
        if not isinstance(mut_ids, torch.Tensor):
            mut_ids = torch.tensor(mut_ids, device=(pf.device))
        mut_ids = mut_ids.to(pf.device)
        append_tensors = batch["append_tensors"]
        if append_tensors.dim() == 3:
            append_tensors = append_tensors[0]
        # Stream A: 浣嶇偣鍋忓ソ
        site_features = pf[mut_ids]
        fitness_21 = self.fitness_mlp(site_features)
        wt_onehot = append_tensors[:, :21]
        mt_onehot = append_tensors[:, 21:]
        fitness_ddg = ((fitness_21 * mt_onehot).sum(-1) - (fitness_21 * wt_onehot).sum(-1)).unsqueeze(-1)
        # Stream B: 缁撴瀯鎵板姩锛堜娇鐢?PairwiseLayer 鑰岄潪 OETRTriadicLayerR1锛?        site_h = self.site_proj(site_features)
        logits_h = self.logits_proj(esm_logits_20[mut_ids])
        wt_h = self.wt_emb(wt_onehot.argmax(-1))
        mt_h = self.mt_emb(mt_onehot.argmax(-1))
        mut_token = self.mut_fuse(torch.cat([site_h, logits_h, wt_h, mt_h], dim=(-1)))
        ctx = self.ctx_proj(pf)
        for layer in self.triadic_layers:
            mut_token = layer(mut_token, ctx, h_e[mut_ids], e_idx[mut_ids], e_mask[mut_ids])

        perturb_score = self.perturb_head(mut_token)
        # Fusion
        ddg = self.ddg_out(torch.cat([fitness_ddg, perturb_score], dim=(-1))).squeeze(-1)
        return ddg

    def forward_mpnn(self, batch, return_graph=False):
        X, S = batch["X"], batch["S"]
        mask, chain_M = batch["mask"], batch["chain_M"]
        residue_idx = batch["residue_idx"]
        chain_encoding_all = batch["chain_encoding_all"]
        all_mpnn_hid, mpnn_embed, _ = self.mpnn_encoder(X, S, mask, chain_M, residue_idx, chain_encoding_all, None, self.use_input_decoding_order)
        all_mpnn_hid = torch.cat([all_mpnn_hid[0], all_mpnn_hid[1], all_mpnn_hid[2], mpnn_embed], dim=(-1))
        if return_graph:
            E, E_idx = self.mpnn_encoder.features(X, mask, residue_idx, chain_encoding_all)
            h_E = self.mpnn_encoder.W_e(E)
            edge_valid_mask = torch.gather(mask.unsqueeze(-1).expand(-1, -1, E_idx.shape[-1]), 1, E_idx).bool()
            edge_valid_mask = edge_valid_mask & mask.unsqueeze(-1).bool()
            return (all_mpnn_hid, E_idx, h_E, edge_valid_mask)
        return all_mpnn_hid

    def get_param_groups(self, lr_backbone=1e-05, lr_fusion=5e-05, lr_head=0.0001):
        backbone_params, fusion_params, head_params = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "mpnn_encoder" in name or "esm_decoder" in name:
                backbone_params.append(param)
            elif "transformer_encoder_protein" in name:
                fusion_params.append(param)
            else:
                head_params.append(param)

        return [
         {'params':backbone_params,
          'lr':lr_backbone},
         {'params':fusion_params,
          'lr':lr_fusion},
         {'params':head_params,
          'lr':lr_head}]
