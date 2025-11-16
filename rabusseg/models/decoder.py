import math
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from .base import ASPP, get_syncbn


class dec_deeplabv3(nn.Module):
    def __init__(
        self,
        in_planes,
        num_classes=19,
        inner_planes=256,
        sync_bn=False,
        dilations=(12, 24, 36),
    ):
        super(dec_deeplabv3, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d

        self.aspp = ASPP(
            in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations
        )
        self.head = nn.Sequential(
            nn.Conv2d(
                self.aspp.get_outplanes(),
                256,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        aspp_out = self.aspp(x)
        res = self.head(aspp_out)
        return res


class dec_deeplabv3_plus(nn.Module):
    def __init__(
        self,
        in_planes,
        num_classes=19,
        inner_planes=256,
        sync_bn=False,
        dilations=(12, 24, 36),
        rep_head=True,
    ):
        super(dec_deeplabv3_plus, self).__init__()
        self.is_corr = True

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.rep_head = rep_head

        self.low_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1), norm_layer(256), nn.ReLU(inplace=True)
        )

        self.aspp = ASPP(
            in_planes, inner_planes=inner_planes, sync_bn=sync_bn, dilations=dilations
        )

        self.head = nn.Sequential(
            nn.Conv2d(
                self.aspp.get_outplanes(),
                256,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

        if self.rep_head:

            self.representation = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
            )

        if self.is_corr:
            self.corr = Corr(nclass=num_classes)
            self.proj = nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
            )

    def forward(self, x, use_corr=True):
        x1, x2, x3, x4 = x
        aspp_out = self.aspp(x4)
        low_feat = self.low_conv(x1)
        aspp_out = self.head(aspp_out)
        # feature = aspp_out
        h, w = low_feat.size()[-2:]
        aspp_out = F.interpolate(
            aspp_out, size=(h, w), mode="bilinear", align_corners=True
        )
        aspp_out = torch.cat((low_feat, aspp_out), dim=1)

        res = {"pred": self.classifier(aspp_out)}

        if self.rep_head:
            res["rep"] = self.representation(aspp_out)

        if use_corr:
            proj_feats = self.proj(x4)
            corr_out_dict = self.corr(proj_feats, res["pred"])
            res['corr_map'] = corr_out_dict['corr_map']
            res['feat_x1'] = x1
            res['feat_x2'] = x2
            res['feat_x3'] = x3
            res['high_feat'] = x4
            corr_out = corr_out_dict['out']
            corr_out = F.interpolate(corr_out, size=(h, w), mode="bilinear", align_corners=True)
            res['corr_out'] = corr_out

        return res


class Aux_Module(nn.Module):
    def __init__(self, in_planes, num_classes=19, sync_bn=False):
        super(Aux_Module, self).__init__()

        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d
        self.aux = nn.Sequential(
            nn.Conv2d(in_planes, 256, kernel_size=3, stride=1, padding=1),
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        res = self.aux(x)
        return res

class Corr(nn.Module):
    def __init__(self, nclass=21):
        super(Corr, self).__init__()
        self.nclass = nclass
        self.kernel_size = 3

    def forward(self, feature_in, out):
        dict_return = {}
        h_in, w_in = math.ceil(feature_in.shape[2] / (1)), math.ceil(feature_in.shape[3] / (1))
        out = F.interpolate(out.detach(), (h_in, w_in), mode='bilinear', align_corners=True)
        feature = rearrange(feature_in, 'n c h w -> n c (h w)')

        # global
        corr_map = torch.matmul(feature.transpose(1, 2), feature) / torch.sqrt(torch.tensor(feature.shape[1]).float())
        dict_return['corr_map'] = corr_map

        # neighbor
        f2_unfold = F.unfold(feature_in, kernel_size=self.kernel_size, padding=self.kernel_size // 2)  # (4,2*9,4225)
        f2_unfold = rearrange(f2_unfold, 'n (c k) (hw) -> n c k (hw)',
                              k=self.kernel_size * self.kernel_size)  # (4, 2*9, 4225)->(4, 2, 9, 4225)
        corr_map_neighbor = torch.matmul(feature.transpose(1, 2).unsqueeze(2), f2_unfold.permute(0, 3, 1, 2)) / torch.sqrt(
            torch.tensor(feature.shape[1]).float())  # (4, 4225, 1, 2)matmul(4, 4225, 2, 9) = (4, 4225, 1, 9)
        corr_map_neighbor = F.softmax(corr_map_neighbor, dim=-1)
        out_unfold = F.unfold(out, kernel_size=self.kernel_size, padding=self.kernel_size // 2)  # (4, 2*9, 4225)
        out_unfold = rearrange(out_unfold, 'n (c k) (hw) -> n c k (hw)',
                               k=self.kernel_size * self.kernel_size)  # (4, 2, 9, 4225)
        dict_return['out'] = rearrange((out_unfold*corr_map_neighbor.permute(0, 2, 3, 1)).sum(dim=2), 'n c (h w) -> n c h w', h=h_in, w=w_in) # (4, 2, 9, 4225)*(4, 1, 9, 4225)=(4, 2, 9, 4225)->(4, 2, 4225)->(4,2,65,65)
        return dict_return

    def sample(self, corr_map, h_in, w_in):
        index = torch.randint(0, h_in * w_in - 1, [128])
        corr_map_sample = corr_map[:, index.long(), :]
        return corr_map_sample

    def normalize_corr_map(self, corr_map, h_in, w_in, h_out, w_out):
        n, m, hw = corr_map.shape
        corr_map = rearrange(corr_map, 'n m (h w) -> (n m) 1 h w', h=h_in, w=w_in)
        corr_map = F.interpolate(corr_map, (h_out, w_out), mode='bilinear', align_corners=True)

        corr_map = rearrange(corr_map, '(n m) 1 h w -> (n m) (h w)', n=n, m=m)
        range_ = torch.max(corr_map, dim=1, keepdim=True)[0] - torch.min(corr_map, dim=1, keepdim=True)[0]
        norm_corr_map = ((- torch.min(corr_map, dim=1, keepdim=True)[0]) + corr_map) / range_
        norm_corr_map = rearrange(norm_corr_map, '(n m) (h w) -> n m (h w)', n=n, m=m, h=h_out, w=w_out)
        return norm_corr_map