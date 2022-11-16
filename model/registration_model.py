from typing import Tuple, Union, List, Optional

import torch
import numpy as np
from monai.losses import DiceLoss, BendingEnergyLoss
from monai.networks import one_hot
from monai.networks.blocks import Warp
from monai.networks.blocks.regunet_block import get_conv_block
from monai.networks.nets import LocalNet
from torch import nn
from torch.nn import functional as F

from data.dataset_utils import organ_index_dict


class Registration(nn.Module):

    def __init__(self, args):
        super(Registration, self).__init__()
        self.args = args
        self.multi_head = args.multi_head
        self.extract_levels = (0, 1, 2, 3)

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

    def forward_output_block(self, output_block, feature_list, image_size):
        layers = output_block.layers
        for layer, level in zip(layers, self.extract_levels):
            print(layer, level)
            print(max(self.extract_levels) - level)
            print("______")
        feature_list = [
            F.interpolate(
                layer(feature_list[max(self.extract_levels) - level]),
                mode="bilinear", size=image_size
            )
            for layer, level in zip(layers, self.extract_levels)
        ]
        out: torch.Tensor = torch.mean(torch.stack(feature_list, dim=0), dim=0)
        return out

    def forward_localnet(self, x):
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
        print(self.model)
        print(len(self.model.decode_deconvs))
        exit()
        for i, (decode_deconv, decode_conv) in enumerate(zip(self.model.decode_deconvs, self.model.decode_convs)):
            print(i)
            # [depth - 1, depth - 2, ..., min_extract_level]
            decoded = decode_deconv(decoded)
            if self.model.concat_skip:
                decoded = torch.cat([decoded, skips[-i - 1]], dim=1)
            else:
                decoded = decoded + skips[-i - 1]
            decoded = decode_conv(decoded)
            outs.append(decoded)

        if self.multi_head:
            ddf_list = [self.forward_output_block(self.output_block_list[i], outs, image_size)
                        for i in range(len(self.output_block_list))]
        else:
            ddf_list = [self.forward_output_block(self.model.output_block, outs, image_size)]

        return ddf_list  # N x (B, 3, H, W, D)

    @staticmethod
    def warp(moving, ddf, one_hot_moving=True, t2w=False):
        """
        :param moving: (B, 1, W, H, D)
        :param ddf: (B, 3, W, H, D)
        :param one_hot_moving: bool, one hot moving before warping
        :param t2w: if input is t2w, warp with "bilinear"
        :return:
        """
        mode = "bilinear" if (one_hot_moving or t2w) else "nearest"
        print(mode)
        pred = Warp(mode="bilinear" if (one_hot_moving or t2w) else "nearest")(
            one_hot(moving, num_classes=9) if one_hot_moving else moving,
            ddf
        )
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
            moving_seg_list, fixed_seg_list = self.separate_seg(moving_seg), self.separate_seg(fixed_seg)  # N x (B, 1, ...)
            loss_organ_list = self.args.organ_list
        else:
            moving_seg_list, fixed_seg_list = [moving_seg], [fixed_seg]  # 1 x (B, 1, ...)
            loss_organ_list = ["all"]
        warped_seg_list = [
            self.warp(ms, ddf, one_hot_moving=self.training)
            for ms, ddf in zip(moving_seg_list, ddf_list)
        ]  # num_class x (B, 9, ...) or num_class x (B, 1, ...)
        if self.training:
            return self.get_loss(warped_seg_list, fixed_seg_list, ddf_list, loss_organ_list)
        else:
            warped_seg = warped_seg_list[0]
            for ws in warped_seg_list[1:]:
                warped_seg += ws * (warped_seg == 0)
            binary = {"seg": warped_seg}
            if not self.multi_head:
                print(ddf_list[0][0, 0])
                warped_t2w = self.warp(moving_batch["t2w"], ddf_list[0], one_hot_moving=False, t2w=True)
                binary["t2w"] = warped_t2w
            return binary

    def get_label_loss(self, warped_seg_list, fixed_seg_list, loss_organ_list):
        """
        :param warped_seg_list: num_class x (B, 9, ...)
        :param fixed_seg_list:
        :param loss_organ_list:
        :return:
        """
        label_loss = {
            f"{organ}_label": self.label_loss(
                ws, fs  # (B, 9, ...), (B, 1, ...)
            )
            for organ, ws, fs in zip(loss_organ_list, warped_seg_list, fixed_seg_list)
        }
        label_loss["label"] = torch.mean(
            torch.stack([v for _, v in label_loss.items()], dim=0), dim=0
        )
        return label_loss

    def get_reg_loss(self, ddf_list):
        if self.args.reg:
            return {
                "reg": 0.1 * torch.mean(
                    torch.tensor([BendingEnergyLoss()(ddf) for ddf in ddf_list])
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

    def separate_seg(self, seg):
        """
        divide a multi-class segmentation into 8 single-class segmentation of the same shape
        :param seg: (B, 1, ...)
        """
        organ_index_list = [organ_index_dict[organ] for organ in self.args.organ_list]
        return [((seg == organ_index) * organ_index).to(seg) for organ_index in organ_index_list]  # 8 x (B, 1, ...)


class RegistrationExtractionBlock(nn.Module):
    """
    The Extraction Block used in RegUNet.
    Extracts feature from each ``extract_levels`` and takes the average.
    """

    def __init__(
        self,
        spatial_dims: int,
        extract_levels: Tuple[int],
        num_channels: Union[Tuple[int], List[int]],
        out_channels: int,
        kernel_initializer: Optional[str] = "kaiming_uniform",
        activation: Optional[str] = None,
    ):
        """

        Args:
            spatial_dims: number of spatial dimensions
            extract_levels: spatial levels to extract feature from, 0 refers to the input scale
            num_channels: number of channels at each scale level,
                List or Tuple of length equals to `depth` of the RegNet
            out_channels: number of output channels
            kernel_initializer: kernel initializer
            activation: kernel activation function
        """
        super().__init__()
        self.extract_levels = extract_levels
        self.max_level = max(extract_levels)
        self.layers = nn.ModuleList(
            [
                get_conv_block(
                    spatial_dims=spatial_dims,
                    in_channels=num_channels[d],
                    out_channels=out_channels,
                    norm=None,
                    act=activation,
                    initializer=kernel_initializer,
                )
                for d in extract_levels
            ]
        )

    def forward(self, x: List[torch.Tensor], image_size: List[int]) -> torch.Tensor:
        """

        Args:
            x: Decoded feature at different spatial levels, sorted from deep to shallow
            image_size: output image size

        Returns:
            Tensor of shape (batch, `out_channels`, size1, size2, size3), where (size1, size2, size3) = ``image_size``
        """
        feature_list = [
            F.interpolate(layer(x[self.max_level - level]), size=image_size)
            for layer, level in zip(self.layers, self.extract_levels)
        ]
        out: torch.Tensor = torch.mean(torch.stack(feature_list, dim=0), dim=0)
        return out