# https://github.com/BytedProtein/ByProt/blob/dd279dc85f76ee2c28c819b71bf3911b90159f0a/src/byprot/models/fixedbb/__init__.py
from omegaconf import OmegaConf
try:
    import esm
    ESM_INSTALLED = True
except:
    ESM_INSTALLED = False

from tristab.utils.config import compose_config, merge_config

import torch
from torch import nn
import numpy as np
import logging

log = logging.getLogger(__name__)

class BaseModel(nn.Module):
    _default_cfg = None

    def __init__(self, cfg) -> None:
        super().__init__()
        self._update_cfg(cfg)

    def _update_cfg(self, cfg):
        if self._default_cfg is None:
            self.cfg = cfg
        else:
            try:
                self.cfg = OmegaConf.merge(self._default_cfg, cfg)
            except Exception as e:
                log.warning(f"OmegaConf merge failed ({e}), filtering unknown keys and retrying")
                # 过滤掉 dataclass 中不存在的 key
                default_keys = set(OmegaConf.to_container(OmegaConf.structured(self._default_cfg)).keys())
                if isinstance(cfg, dict):
                    filtered_cfg = {k: v for k, v in cfg.items() if k in default_keys}
                else:
                    filtered_cfg = {k: v for k, v in OmegaConf.to_container(cfg).items() if k in default_keys}
                try:
                    self.cfg = OmegaConf.merge(self._default_cfg, filtered_cfg)
                except Exception as e2:
                    log.error(f"Retry also failed: {e2}")
                    raise e2

    @classmethod
    def from_config(cls, cfg):
        raise NotImplementedError

    def forward_encoder(self, batch):
        raise NotImplementedError

    def forward_decoder(self, prev_decoder_out, encoder_out):
        raise NotImplementedError

    def initialize_output_tokens(self, batch, encoder_out):
        raise NotImplementedError

    def forward(self, coords, coord_mask, tokens, token_padding_mask=None, **kwargs):
        raise NotImplementedError

    def sample(self, coords, coord_mask, tokens=None, token_padding_mask=None, **kwargs):
        raise NotImplementedError
