import torch
import numpy as np
from monai.losses import DiceLoss, BendingEnergyLoss
from monai.networks import one_hot
from monai.networks.blocks import Warp
from monai.networks.nets import LocalNet
from torch import nn
import torch.nn.functional as F

from data.dataset_utils import organ_index_dict


class PatchRegistration(nn.Module):

    def __init__(self, args):
        super(PatchRegistration, self).__init__()
        self.args = args
        self.multi_head = args.multi_head

        self.model = LocalNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=3,
            num_channel_initial=32,
            extract_levels=(0, 1, 2, 3),
            out_activation=None,
            out_kernel_initializer="zeros"
        )

        if self.multi_head:
            self.output_block_list = nn.ModuleList(
                [self.model.output_block] + [self.model.build_output_block() for _ in range(len(args.organ_list) - 1)]
            )

        # self.img_loss = GlobalMutualInformationLoss()
        self.img_loss = nn.MSELoss()
        self.label_loss = DiceLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
        )
        self.regularisation = BendingEnergyLoss()

    def forward_localnet(self, x):
        if self.multi_head:
            image_size = x.shape[2:]
            skips = []  # [0, ..., depth - 1]
            encoded = x
            for encode_conv, encode_pool in zip(self.model.encode_convs, self.model.encode_pools):
                skip = encode_conv(encoded)
                encoded = encode_pool(skip)
                skips.append(skip)
            decoded = self.model.bottom_block(encoded)

            outs = [decoded]

            # [depth - 1, ..., min_extract_level]
            for i, (decode_deconv, decode_conv) in enumerate(zip(self.model.decode_deconvs, self.model.decode_convs)):
                # [depth - 1, depth - 2, ..., min_extract_level]
                decoded = decode_deconv(decoded)
                if self.model.concat_skip:
                    decoded = torch.cat([decoded, skips[-i - 1]], dim=1)
                else:
                    decoded = decoded + skips[-i - 1]
                decoded = decode_conv(decoded)
                outs.append(decoded)

            ddf_list = [self.output_block_list[i](outs, image_size=image_size)
                        for i in range(len(self.output_block_list))]

            return ddf_list  # N x (B, 3, H, W, D)
        else:
            return [self.model(x)]  # 1 x (B, 3, H, W, D)

    @staticmethod
    def warp(moving, ddf, one_hot_moving=True):
        """
        :param moving: (B, 1, W, H, D)
        :param ddf: (B, 3, W, H, D)
        :param one_hot_moving: bool, one hot moving before warping
        :return:
        """
        pred = Warp(mode="bilinear" if one_hot_moving else "nearest")(
            one_hot(moving, num_classes=9) if one_hot_moving else moving,
            ddf
        )
        return pred  # (B, 9, ...) or (B, 1, ...)

    def forward(self, moving_batch, fixed_batch):
        """
        :param moving_batch:
        "t2w": (B, 1, ...)
        "seg": (B, 1, ...)
        "pos": (B, 3)
        "name": str
        "ins": int
        :param fixed_batch:
        "t2w": (B, 1, ...)
        "seg": (B, 1, ...)
        "pos": (B, 3)
        "name": str
        "ins": int
        :return:
        """

        if self.args.input == "img":
            x = torch.cat([moving_batch["t2w"], fixed_batch["t2w"]], dim=1)
        else:
            x = torch.cat([moving_batch["seg"], fixed_batch["seg"]], dim=1)
        ddf_list = self.forward_localnet(x)  # num_class x (B, 3, H, W, D)
        ddf = torch.stack(ddf_list, dim=-1)  # (B, 3, ..., num_class)
        ddf = torch.mean(ddf, dim=(-1, -2, -3, -4))  # (B, 3)
        loss = self.get_loss(ddf, moving_batch, fixed_batch)
        if self.training:
            return loss
        else:
            return ddf, loss

    def get_loss(self, ddf, moving_batch, fixed_batch):
        """
        :param ddf: (B, 3)
        :param moving_batch:
        :param fixed_batch:
        :return:
        """
        loss = F.mse_loss(
            ddf,
            (fixed_batch["pos"] - moving_batch["pos"]).to(ddf)
        )
        return {"total": loss}
