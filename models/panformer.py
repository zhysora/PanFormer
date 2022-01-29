import torch
from torch import nn

from .common.modules import conv3x3, SwinModule
from .base_model import Base_model
from .builder import MODELS


class CrossSwinTransformer(nn.Module):
    def __init__(self, cfg, logger, n_feats=64, n_heads=4, head_dim=16, win_size=4,
                 n_blocks=3, cross_module=['pan', 'ms'], cat_feat=['pan', 'ms'], sa_fusion=False):
        super().__init__()
        self.cfg = cfg
        self.n_blocks = n_blocks
        self.cross_module = cross_module
        self.cat_feat = cat_feat
        self.sa_fusion = sa_fusion

        pan_encoder = [
            SwinModule(in_channels=1, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=2, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
            SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=2, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
        ]
        ms_encoder = [
            SwinModule(in_channels=cfg.ms_chans, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
            SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                       downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
        ]

        if 'ms' in self.cross_module:
            self.ms_cross_pan = nn.ModuleList()
            for _ in range(n_blocks):
                self.ms_cross_pan.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                                    downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                                                    window_size=win_size, relative_pos_embedding=True, cross_attn=True))
        elif sa_fusion:
            ms_encoder.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                         downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                                         window_size=win_size, relative_pos_embedding=True, cross_attn=False))

        if 'pan' in self.cross_module:
            self.pan_cross_ms = nn.ModuleList()
            for _ in range(n_blocks):
                self.pan_cross_ms.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                                    downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                                                    window_size=win_size, relative_pos_embedding=True, cross_attn=True))
        elif sa_fusion:
            pan_encoder.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                          downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                                          window_size=win_size, relative_pos_embedding=True, cross_attn=False))

        self.HR_tail = nn.Sequential(
            conv3x3(n_feats * len(cat_feat), n_feats * 4),
            nn.PixelShuffle(2), nn.ReLU(True), conv3x3(n_feats, n_feats * 4),
            nn.PixelShuffle(2), nn.ReLU(True), conv3x3(n_feats, n_feats),
            nn.ReLU(True), conv3x3(n_feats, cfg.ms_chans))

        self.pan_encoder = nn.Sequential(*pan_encoder)
        self.ms_encoder = nn.Sequential(*ms_encoder)

    def forward(self, pan, ms):
        pan_feat = self.pan_encoder(pan)
        ms_feat = self.ms_encoder(ms)

        last_pan_feat = pan_feat
        last_ms_feat = ms_feat
        for i in range(self.n_blocks):
            if 'pan' in self.cross_module:
                pan_cross_ms_feat = self.pan_cross_ms[i](last_pan_feat, last_ms_feat)
            if 'ms' in self.cross_module:
                ms_cross_pan_feat = self.ms_cross_pan[i](last_ms_feat, last_pan_feat)
            if 'pan' in self.cross_module:
                last_pan_feat = pan_cross_ms_feat
            if 'ms' in self.cross_module:
                last_ms_feat = ms_cross_pan_feat

        cat_list = []
        if 'pan' in self.cat_feat:
            cat_list.append(last_pan_feat)
        if 'ms' in self.cat_feat:
            cat_list.append(last_ms_feat)

        output = self.HR_tail(torch.cat(cat_list, dim=1))

        if self.cfg.norm_input:
            output = torch.clamp(output, 0, 1)
        else:
            output = torch.clamp(output, 0, 2 ** self.cfg.bit_depth - .5)

        return output


@MODELS.register_module()
class PanFormer(Base_model):
    def __init__(self, cfg, logger, train_data_loader, test_data_loader0, test_data_loader1):
        super().__init__(cfg, logger, train_data_loader, test_data_loader0, test_data_loader1)

        model_cfg = cfg.get('model_cfg', dict())
        G_cfg = model_cfg.get('core_module', dict())

        self.add_module('core_module', CrossSwinTransformer(cfg=cfg, logger=logger, **G_cfg))

    def get_model_output(self, input_batch):
        input_pan = input_batch['input_pan']
        input_lr = input_batch['input_lr']
        output = self.module_dict['core_module'](input_pan, input_lr)
        return output

    def train_iter(self, iter_id, input_batch, log_freq=10):
        G = self.module_dict['core_module']
        G_optim = self.optim_dict['core_module']

        input_pan = input_batch['input_pan']
        input_lr = input_batch['input_lr']

        output = G(input_pan, input_lr)

        loss_g = 0
        loss_res = dict()
        loss_cfg = self.cfg.get('loss_cfg', {})
        if 'rec_loss' in self.loss_module:
            target = input_batch['target']
            rec_loss = self.loss_module['rec_loss'](
                out=output, gt=target
            )
            loss_g = loss_g + rec_loss * loss_cfg['rec_loss'].w
            loss_res['rec_loss'] = rec_loss.item()

        loss_res['full_loss'] = loss_g.item()

        G_optim.zero_grad()
        loss_g.backward()
        G_optim.step()

        self.print_train_log(iter_id, loss_res, log_freq)
