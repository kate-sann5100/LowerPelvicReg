import torch
import numpy as np
from monai.losses import DiceLoss, BendingEnergyLoss
from monai.networks import one_hot
from monai.networks.blocks import Warp
from monai.networks.nets import LocalNet
from torch import nn

from data.dataset_utils import organ_list


class Registration(nn.Module):

    def __init__(self, args):
        super(Registration, self).__init__()
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
                [self.model.bottom_block] + [self.model.build_output_block() for _ in range(7)]
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

            ddf_list = [self.output_block_list[i](outs, image_size=image_size) for i in range(8)]
            return ddf_list  # num_class x (B, 3, H, W, D)
        else:
            return [self.model(x)]  # 1 x (B, 3, H, W, D)

    @staticmethod
    def warp(moving, ddf, binary=False):
        """
        :param moving: (B, 1, W, H, D)
        :param ddf: (B, 3, W, H, D)
        :param binary: if moving is binary
        :return:
        """
        print(f"moving shape in self.warp: {moving.shape}")
        if not binary:
            moving = one_hot(moving, num_classes=9)  # (B, 9, ...)
        pred = Warp(mode="nearest" if binary else "bilinear")(moving, ddf)
        return pred  # (B, 9, ...) or (B, 1, ...)

    def forward(self, moving_batch, fixed_batch):
        """
        :param moving_batch:
        "t2w": (B, 1, ...)
        "seg": (B, 1, ...)
        "name": str
        "ins": int
        :param fixed_batch:
        "t2w": (B, 1, ...)
        "seg": (B, 1, ...)
        "name": str
        "ins": int
        :return:
        """

        if self.args.input == "img":
            x = torch.cat([moving_batch["t2w"], fixed_batch["t2w"]], dim=1)
        else:
            x = torch.cat([moving_batch["seg"], fixed_batch["seg"]], dim=1)
        ddf_list = self.forward_localnet(x)  # num_class x (B, 3, H, W, D)
        moving_seg, fixed_seg = moving_batch["seg"], fixed_batch["seg"]  # (B, 1, ...)
        if self.multi_head:
            moving_seg_list, fixed_seg_list = separate_seg(moving_seg), separate_seg(fixed_seg)  # 9 x (B, 1, ...)
            loss_organ_list = organ_list
        else:
            moving_seg_list, fixed_seg_list = [moving_seg], [fixed_seg]  # 1 x (B, 1, ...)
            loss_organ_list = ["all"]
        print([ms.shape for ms in moving_seg_list])
        warped_seg_list = [
            self.warp(ms, ddf, binary=not self.training)
            for ms, ddf in zip(moving_seg, ddf_list)
        ]  # num_class x (B, 9, ...) or num_class x (B, 1, ...)
        if self.training:
            return self.get_loss(warped_seg_list, fixed_seg_list, ddf_list, loss_organ_list)
        else:
            warped_seg = torch.sum(
                # num_class x (B, 1, ...) -> (B, 1, ..., num_class)
                torch.stack(warped_seg_list, dim=-1), dim=-1
            )  # (B, 1, ...)
            return warped_seg

    def get_label_loss(self, warped_seg_list, fixed_seg_list, loss_organ_list):
        """
        :param warped_seg_list: num_class x (B, 9, ...)
        :param fixed_seg_list:
        :param loss_organ_list:
        :return:
        """
        label_loss = {
            f"{organ}_label": self.label_loss(
                one_hot(ws, num_classes=9), fs  # (B, 9, ...), (B, 1, ...)
            )
            for organ, ws, fs in zip(loss_organ_list, warped_seg_list, fixed_seg_list)
        }
        label_loss["label"] = torch.mean(
            torch.tensor([v for _, v in label_loss.items()])
        )
        return label_loss

    def get_reg_loss(self, ddf_list):
        if self.args.reg:
            return {
                "reg": torch.mean(
                    torch.tensor([BendingEnergyLoss(ddf) for ddf in ddf_list])
                )
            }
        else:
            return None

    def get_loss(self, warped_seg_list, fixed_seg_list, ddf_list, loss_organ_list):
        loss_dict = {}
        label_loss = self.get_label_loss(warped_seg_list, fixed_seg_list, loss_organ_list)  # n x scalar
        loss_dict.update(label_loss)
        reg_loss = self.get_reg_loss(ddf_list)
        if reg_loss is None:
            reg_loss = {"reg": torch.zeros_like(label_loss["label"])}
        loss_dict.update(reg_loss)
        return loss_dict


def separate_seg(seg):
    """
    divide a multi-class segmentation into 8 single-class segmentation of the same shape
    :param seg: (B, 1, ...)
    """
    seg = [seg] * 8
    for i in range(8):
        seg[i][seg[i] != i + 1] = 0
    return seg  # 8 x (B, 1, H, W, D)
