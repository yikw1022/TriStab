
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


# -----------------------------------------------------------------------------
# Panel (c): a single layer of the overlap-aware local-context
# residue-pair perturbation module.
# -----------------------------------------------------------------------------
class LocalPerturbationLayer(nn.Module):
   
    def __init__(self, d_model: int, d_edge: int, K: int = 4,
                 dropout: float = 0.5):
        super().__init__()
        self.K = K
        self.d = d_model

        self.edge_proj = nn.Linear(d_edge, d_model)

        # Pair-message MLP:  R^{5d+1} 鈫?R^{d}
        self.message_mlp = nn.Sequential(
            nn.Linear(d_model * 5 + 1, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        # Pair-weight MLP:   R^{5d+1} 鈫?R^{1}   (pre-softmax logits)
        self.pair_weight_mlp = nn.Sequential(
            nn.Linear(d_model * 5 + 1, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        # Gating function for the residual update.
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, center_repr, h_ctx, edge_feat, neighbors, nb_mask):
       
        K = min(self.K, neighbors.shape[1])
        M, d = center_repr.shape

        nb_idx = neighbors[:, :K]                       # (M, K)
        mask = nb_mask[:, :K]                           # (M, K)

        H_N = h_ctx[nb_idx]                             # (M, K, d)
        H_E = self.edge_proj(edge_feat[:, :K])          # (M, K, d)

        # Broadcast to (M, K, K, d): one entry per ordered pair (u, v).
        h_u = H_N.unsqueeze(2).expand(-1, -1, K, -1)
        h_v = H_N.unsqueeze(1).expand(-1, K, -1, -1)
        h_c = center_repr.unsqueeze(1).unsqueeze(1).expand(-1, K, K, -1)
        e_cu = H_E.unsqueeze(2).expand(-1, -1, K, -1)
        e_cv = H_E.unsqueeze(1).expand(-1, K, -1, -1)

        # Pairwise cosine similarity s_{u,v} between context residues.
        H_N_norm = F.normalize(H_N, dim=-1)
        sim = torch.bmm(H_N_norm, H_N_norm.transpose(1, 2)).unsqueeze(-1)

        # Mask out u==v and padded neighbors.
        diag_mask = ~torch.eye(K, dtype=torch.bool, device=center_repr.device)
        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1) & diag_mask.unsqueeze(0)

        # Shared triadic input fed into both MLPs.
        pair_input = torch.cat([h_c, h_u, h_v, e_cu, e_cv, sim], dim=-1)

        # Pair weights 伪_{u,v}: softmax over all valid (u, v) pairs.
        alpha_logits = self.pair_weight_mlp(pair_input).squeeze(-1)
        alpha_logits = alpha_logits.masked_fill(~pair_mask, -1e4)
        alpha = torch.softmax(alpha_logits.view(M, -1), dim=-1).view(M, K, K)

        # Per-pair messages m_{u,v} and aggregated message m_c.
        pair_msg = self.message_mlp(pair_input)                  # (M, K, K, d)
        msg = (pair_msg * alpha.unsqueeze(-1)).sum(dim=(1, 2))    # (M, d)

        # Gated residual update of the center representation.
        g = self.gate(torch.cat([center_repr, msg], dim=-1))
        return self.layer_norm(center_repr + g * msg)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class TriStabConfig:
    encoder_mpnn: ProteinMPNNConfig = field(default_factory=ProteinMPNNConfig)
    esm_name: str = "esm2_t33_650M_UR50D"
    mpnn_name: str = "ProteinMPNN"
    dropout: float = 0.1
    esm_tune: bool = False
    mpnn_tune: bool = True

    # Local-perturbation branch hyperparameters (paper notation).
    d_model: int = 256            # d   鈥?hidden dim
    d_edge: int = 128             # d_E 鈥?edge feature dim
    K: int = 4                    # K   鈥?number of local-context residues
    n_perturb_layers: int = 2     # L_perturb 鈥?stacked perturbation layers
    perturb_dropout: float = 0.5  # p_perturb 鈥?dropout inside the MLPs


# -----------------------------------------------------------------------------
# TriStab top-level model
# -----------------------------------------------------------------------------
@register_model("tristab")
class TriStab(BaseModel):

    _default_cfg = TriStabConfig()

    def __init__(self, cfg):
        super().__init__(cfg)

        # ---- Pretrained encoders (shared across all candidate mutations) ----
        self.esm_decoder = ESM2.from_pretrained(
            args=self.cfg, name=self.cfg.esm_name,
        )
        self.padding_idx = self.esm_decoder.padding_idx
        self.mpnn_encoder = get_protein_mpnn(tune=cfg.mpnn_tune)
        self.use_input_decoding_order = cfg.encoder_mpnn.use_input_decoding_order

        # ---- Sequence+structure fusion encoder over (ESM 鈯?MPNN) features ---
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=1792, nhead=8, dropout=0.1,
            dim_feedforward=3584, batch_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(fusion_layer, num_layers=2)

        d = getattr(self.cfg, "d_model", 256)
        d_edge = getattr(self.cfg, "d_edge", 128)
        K = getattr(self.cfg, "K", 4)
        n_perturb_layers = getattr(self.cfg, "n_perturb_layers", 2)
        perturb_dropout = getattr(self.cfg, "perturb_dropout", 0.5)

        # ---- Site-level substitution branch (produces 未_1) ------------------
        self.substitution_mlp = nn.Sequential(
            nn.Linear(1792, 1024), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(1024, 21),
        )

        # ---- Center-representation construction -----------------------------
        self.site_proj = nn.Linear(1792, d)        # h_site projection
        self.ell_proj = nn.Linear(20, d)           # 鈩揰i (ESM2 per-position log-prob)
        self.wt_emb = nn.Embedding(21, d)          # WT amino-acid embedding
        self.mt_emb = nn.Embedding(21, d)          # MT amino-acid embedding
        self.center_fuse = nn.Sequential(
            nn.Linear(d * 4, d * 2), nn.GELU(),
            nn.Linear(d * 2, d),
        )

        # ---- Local perturbation branch (produces 未_2) -----------------------
        self.ctx_proj = nn.Linear(1792, d)         # h_ctx projection
        self.perturb_layers = nn.ModuleList([
            LocalPerturbationLayer(d, d_edge, K=K, dropout=perturb_dropout)
            for _ in range(n_perturb_layers)
        ])
        self.perturb_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d // 2, 1),
        )

        # ---- Output head: 螖螖臏 = output_head([未_1, 未_2]) ---------------------
        self.output_head = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(16, 1),
        )

    # -------------------------------------------------------------------------
    def forward(self, batch, **kwargs):
        # Encode WT sequence + structure once; reuse across candidate mutations.
        with torch.set_grad_enabled(self.cfg.mpnn_tune):
            mpnn_feats, E_idx, h_E, edge_valid_mask = self.forward_mpnn(
                batch, return_graph=True,
            )
        with torch.set_grad_enabled(self.cfg.esm_tune):
            esm_out = self.esm_decoder(tokens=batch["tokens"], encoder_out=None)
            esm_feats = esm_out["representations"][-1][:, 1:-1]
            esm_logits = esm_out["logits"][:, 1:-1]

        # Fused per-residue protein feature (length L, dim 1792).
        h_prot = self.fusion_encoder(
            torch.cat([esm_feats, mpnn_feats], dim=-1),
        )[0]
        h_e = h_E[0] if h_E.dim() == 4 else h_E
        e_idx = E_idx[0] if E_idx.dim() == 3 else E_idx
        e_mask = edge_valid_mask[0] if edge_valid_mask.dim() == 3 else edge_valid_mask

        # 鈩揰i 鈭?R^{20}: ESM2 per-position log-probability over the 20 AAs.
        # (Indices 4:24 select the 20 standard amino-acid tokens.)
        ell = F.log_softmax(esm_logits[0], dim=-1)[:, 4:24]

        # Mutation-site indices.
        mut_ids = batch["mut_ids"]
        if not isinstance(mut_ids, torch.Tensor):
            mut_ids = torch.tensor(mut_ids, device=h_prot.device)
        mut_ids = mut_ids.to(h_prot.device)

        append_tensors = batch["append_tensors"]
        if append_tensors.dim() == 3:
            append_tensors = append_tensors[0]
        wt_onehot = append_tensors[:, :21]
        mt_onehot = append_tensors[:, 21:]

        # ---- 未_1: site-level substitution branch ---------------------------
        h_site = h_prot[mut_ids]
        subs_logits_21 = self.substitution_mlp(h_site)
        delta_1 = (
            (subs_logits_21 * mt_onehot).sum(-1)
            - (subs_logits_21 * wt_onehot).sum(-1)
        ).unsqueeze(-1)

        # ---- 未_2: local perturbation branch --------------------------------
        # Mutation-conditioned center representation: fuse (h_site, 鈩揰i, h_wt, h_mt).
        center_repr = self.center_fuse(torch.cat([
            self.site_proj(h_site),
            self.ell_proj(ell[mut_ids]),
            self.wt_emb(wt_onehot.argmax(-1)),
            self.mt_emb(mt_onehot.argmax(-1)),
        ], dim=-1))

        h_ctx = self.ctx_proj(h_prot)
        for layer in self.perturb_layers:
            center_repr = layer(
                center_repr, h_ctx,
                h_e[mut_ids], e_idx[mut_ids], e_mask[mut_ids],
            )
        delta_2 = self.perturb_head(center_repr)

        # ---- Output head ---------------------------------------------------
        ddg_hat = self.output_head(
            torch.cat([delta_1, delta_2], dim=-1),
        ).squeeze(-1)
        return ddg_hat

    # -------------------------------------------------------------------------
    def forward_mpnn(self, batch, return_graph=False):
        X, S = batch["X"], batch["S"]
        mask, chain_M = batch["mask"], batch["chain_M"]
        residue_idx = batch["residue_idx"]
        chain_encoding_all = batch["chain_encoding_all"]

        all_mpnn_hid, mpnn_embed, _ = self.mpnn_encoder(
            X, S, mask, chain_M, residue_idx, chain_encoding_all,
            None, self.use_input_decoding_order,
        )
        all_mpnn_hid = torch.cat(
            [all_mpnn_hid[0], all_mpnn_hid[1], all_mpnn_hid[2], mpnn_embed],
            dim=-1,
        )
        if return_graph:
            E, E_idx = self.mpnn_encoder.features(
                X, mask, residue_idx, chain_encoding_all,
            )
            h_E = self.mpnn_encoder.W_e(E)
            edge_valid_mask = torch.gather(
                mask.unsqueeze(-1).expand(-1, -1, E_idx.shape[-1]), 1, E_idx,
            ).bool()
            edge_valid_mask = edge_valid_mask & mask.unsqueeze(-1).bool()
            return all_mpnn_hid, E_idx, h_E, edge_valid_mask
        return all_mpnn_hid

    # -------------------------------------------------------------------------
    def get_param_groups(self, lr_backbone=1e-5, lr_fusion=5e-5, lr_head=1e-4):
        backbone, fusion, head = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "mpnn_encoder" in name or "esm_decoder" in name:
                backbone.append(param)
            elif "fusion_encoder" in name:
                fusion.append(param)
            else:
                head.append(param)
        return [
            {"params": backbone, "lr": lr_backbone},
            {"params": fusion,   "lr": lr_fusion},
            {"params": head,     "lr": lr_head},
        ]



def migrate_checkpoint(old_state_dict: dict) -> dict:
   
    rename_rules = [
        ("transformer_encoder_protein.", "fusion_encoder."),
        ("fitness_mlp.",                  "substitution_mlp."),
        ("logits_proj.",                  "ell_proj."),
        ("mut_fuse.",                     "center_fuse."),
        ("ddg_out.",                      "output_head."),
        # Stack of perturbation layers.
        ("triadic_layers.",               "perturb_layers."),
        # Submodules inside each LocalPerturbationLayer.
        (".triadic_fn.",                  ".message_mlp."),
        (".pair_score.",                  ".pair_weight_mlp."),
        # `self.norm` inside the layer 鈫?`self.layer_norm` (scoped to
        # avoid touching any external `*.norm.*` if present).
        ("perturb_layers.",               "perturb_layers."),  # no-op anchor
    ]

    new_state_dict = {}
    for k, v in old_state_dict.items():
        nk = k
        for old, new in rename_rules:
            if old in nk:
                nk = nk.replace(old, new)
        # Layer-scoped `.norm.` 鈫?`.layer_norm.` rename, restricted to
        # parameters under the perturbation-layer stack.
        if nk.startswith("perturb_layers.") and ".norm." in nk:
            nk = nk.replace(".norm.", ".layer_norm.")
        new_state_dict[nk] = v
    return new_state_dict