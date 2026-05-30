"""
娑堣瀺瀹為獙 A1: w/o 缁撴瀯鎵板姩鍒嗘敮

鐩告瘮瀹屾暣妯″瀷 (hyperppn_lite_oetr_r1)锛?  - 绉婚櫎鏁翠釜 Stream B锛堜笁浣撳眰 + perturb_head + 绐佸彉鏉′欢琛ㄧず鏋勯€狅級
  - 浠呬繚鐣欎綅鐐瑰亸濂藉垎鏀?(fitness_ddg)
  - 绉婚櫎 ddg_out 铻嶅悎灞傦紝鐩存帴杈撳嚭 fitness_ddg
  - 楠岃瘉涓変綋缁撴瀯寤烘ā鏄惁蹇呰

娉ㄥ唽涓?'ablation_no_perturb'
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


@dataclass
class ConfigAblationNoPerturb:
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


@register_model("ablation_no_perturb")
class AblationNoPerturb(BaseModel):
    _default_cfg = ConfigAblationNoPerturb()

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
        # ========== 浠呬繚鐣?Stream A: 浣嶇偣鍋忓ソ MLP ==========
        self.fitness_mlp = nn.Sequential(nn.Linear(1792, 1024), nn.GELU(), nn.Dropout(0.2), nn.Linear(1024, 21))
        # 銆愭秷铻嶃€戠Щ闄や簡 Stream B 鐨勬墍鏈夌粍浠讹細
        #   site_proj, logits_proj, wt_emb, mt_emb, mut_fuse,
        #   ctx_proj, triadic_layers, perturb_head
        # 銆愭秷铻嶃€戠Щ闄や簡 ddg_out 铻嶅悎灞傦紝鐩存帴杈撳嚭 fitness_ddg

    def forward(self, batch, **kwargs):
        # 銆愭秷铻嶃€戜笉闇€瑕佸浘淇℃伅锛坔_E, E_idx锛夛紝鐢?return_graph=False
        with torch.set_grad_enabled(self.cfg.mpnn_tune):
            mpnn_features = self.forward_mpnn(batch, return_graph=False)
        with torch.set_grad_enabled(self.cfg.esm_tune):
            esm_out = self.esm_decoder(tokens=(batch["tokens"]), encoder_out=None)
            esm_features = esm_out["representations"][-1][:, 1:-1]
        protein_feature = torch.cat([esm_features, mpnn_features], dim=(-1))
        protein_feature = self.transformer_encoder_protein(protein_feature)
        pf = protein_feature[0]
        mut_ids = batch["mut_ids"]
        if not isinstance(mut_ids, torch.Tensor):
            mut_ids = torch.tensor(mut_ids, device=(pf.device))
        mut_ids = mut_ids.to(pf.device)
        append_tensors = batch["append_tensors"]
        if append_tensors.dim() == 3:
            append_tensors = append_tensors[0]
        # ========== Stream A only ==========
        site_features = pf[mut_ids]
        fitness_21 = self.fitness_mlp(site_features)
        wt_onehot = append_tensors[:, :21]
        mt_onehot = append_tensors[:, 21:]
        # 銆愭秷铻嶃€戠洿鎺ヨ緭鍑?fitness_ddg锛屼笉缁忚繃铻嶅悎灞?        fitness_ddg = (fitness_21 * mt_onehot).sum(-1) - (fitness_21 * wt_onehot).sum(-1)
        return fitness_ddg

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
