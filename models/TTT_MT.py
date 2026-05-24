import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange as o_rearrange
INTERPOLATE_MODE = 'bilinear'
BATCHNORM = nn.SyncBatchNorm # nn.BatchNorm2d


def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()


class ConvHead_inter(nn.Module):
    def __init__(self, in_channels, intermediate_channel, out_channels, momentum=0.1):
        super(ConvHead_inter, self).__init__()
        self.intermediate = nn.Conv2d(in_channels, intermediate_channel, kernel_size=1)
        self.bn = BATCHNORM(intermediate_channel, momentum=momentum)
        self.linear_pred = nn.Conv2d(intermediate_channel, out_channels, kernel_size=1)

    def forward(self, x):
        x_inter = self.bn(self.intermediate(x))
        out = self.linear_pred(x_inter)
        return out, x_inter


class S4T_MAE_v4(nn.Module):
    """ S4T model.

    The backbone features are compressed into per-task feature maps (``ts_comp``)
    that feed (i) auxiliary task heads and (ii) a joint MAE-style decoder which
    reconstructs all task outputs from a masked input. The reconstruction branch
    provides the self-supervised signal used at test time.
    """
    def __init__(self, p, backbone, backbone_channels, heads, S4T_decoder, aux_heads):
        super(S4T_MAE_v4, self).__init__()
        self.p = p
        self.tasks = p.TASKS.NAMES
        if 'TASKS_all' in p.keys():
            self.tasks_all = p.TASKS_all.NAMES
            self.num_out = p.TASKS_all.NUM_OUTPUT
        else:
            self.tasks_all = self.tasks
            self.num_out = p.TASKS.NUM_OUTPUT
        self.backbone = backbone
        self.heads = heads
        self.joint_dec = S4T_decoder

        self.ch = p.head_inter_channel
        if 'TASKS_all' in p.keys():
            self.ts_comp = nn.Conv2d(sum(backbone_channels), len(self.tasks_all)*self.ch, kernel_size=1)
        else:
            if 'backbone_layer' in p.keys():
                n_cha = sum(backbone_channels[i] for i in self.p.backbone_layer)
            else:
                n_cha = sum(backbone_channels)
            self.ts_comp = nn.Conv2d(n_cha, len(self.tasks)*self.ch, kernel_size=1)
        self.aux_heads = aux_heads
        if 'mask_ratio' in p.S4T_model_spec.keys():
            self.mask_ratio = p.S4T_model_spec.mask_ratio
        else:
            raise ValueError("S4T_model_spec.mask_ratio must be specified.")

    def forward(self, x):
        img_size = x.size()[-2:]

        # Backbone
        x, selected_fea = self.backbone(x)

        _x_list = []
        if 'backbone_layer' in self.p.keys():
            _x_list_TTT = []
        for i, fea in enumerate(selected_fea):
            if self.p.model == 'CNN_MTL_S4T':
                _x = F.interpolate(fea, self.p.spatial_dim[0], mode=INTERPOLATE_MODE)
            else:
                oh, ow = self.p.spatial_dim[0]
                _x = rearrange(fea, 'b (h w) c -> b c h w', h=oh, w=ow)
            if 'backbone_layer' in self.p.keys():
                if i in self.p.backbone_layer:
                    _x_list_TTT.append(_x)
            _x_list.append(_x)

        _x = torch.cat(_x_list, dim=1)
        if 'backbone_layer' in self.p.keys():
            _x_TTT = torch.cat(_x_list_TTT, dim=1)
        else:
            _x_TTT = _x

        # Generate preliminary predictions
        concat_fea = self.ts_comp(_x_TTT)
        out = {}; out_aux = {}
        for i, t in enumerate(self.tasks):
            _out = self.heads[t](_x)
            out[t] = F.interpolate(_out, img_size, mode=INTERPOLATE_MODE)
        for i, t in enumerate(self.tasks_all):
            _out_aux = self.aux_heads[t](concat_fea[:,self.ch*i:self.ch*(i+1)])
            out_aux[t] = F.interpolate(_out_aux, img_size, mode=INTERPOLATE_MODE)

        TTT_in = F.interpolate(concat_fea, self.p.S4T_model_spec.TTT_input_res, mode=INTERPOLATE_MODE)

        TTT_out = {}
        start_channel, end_channel = 0, 0
        _TTT_out = self.joint_dec.unpatchify(self.joint_dec(TTT_in, self.mask_ratio))
        for t in self.tasks_all:
            end_channel += self.num_out[t]
            TTT_out[t] = F.interpolate(_TTT_out[:,start_channel:end_channel], img_size, mode=INTERPOLATE_MODE)
            start_channel = end_channel

        return out, TTT_out, out_aux
