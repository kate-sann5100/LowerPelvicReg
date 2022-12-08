from typing import Tuple, Union, List, Optional

import torch
import numpy as np
from monai.losses import DiceLoss, BendingEnergyLoss
from monai.networks import one_hot
from monai.networks.blocks import Warp
from monai.networks.blocks.regunet_block import get_conv_block, get_deconv_block
from monai.networks.nets import LocalNet, RegUNet
from torch import nn
from torch.nn import functional as F, MSELoss

from data.dataset_utils import organ_index_dict


class Registration(nn.Module):

    def __init__(self, args):
        super(Registration, self).__init__()
        self.args = args
        self.multi_head = args.multi_head
        self.extract_levels = (0, 1, 2, 3)

        self.model = NewLocalNet(
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
        feature_list = [
            F.interpolate(
                layer(feature_list[max(self.extract_levels) - level]),
                mode="trilinear", size=image_size
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
        for i, (decode_deconv, decode_conv) in enumerate(zip(self.model.decode_deconvs, self.model.decode_convs)):
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
        pred = Warp(mode="bilinear" if (one_hot_moving or t2w) else "nearest")(
            one_hot(moving, num_classes=9) if one_hot_moving else moving,
            ddf
        )
        return pred  # (B, 9, ...) or (B, 1, ...)

    def forward(self, moving_batch, fixed_batch,
                semi_supervision=False, semi_mode=None):
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
        :param semi_supervision: bool
        :param semi_mode: "train" or "eval"
        :return:
        """

        if self.args.input == "img":
            x = torch.cat([moving_batch["t2w"], fixed_batch["t2w"]], dim=1)
        else:
            x = torch.cat([moving_batch["seg"], fixed_batch["seg"]], dim=1)
        ddf_list = self.forward_localnet(x)  # num_class x (B, 3, H, W, D)

        if semi_supervision and semi_mode == "train":
            ddf = torch.mean(
                torch.stack(ddf_list, dim=-1),  # (B, 3, H, W, D, num_class)
                dim=-1
            )  # (B, 3, H, W, D)
            print("line136")
            return ddf

        moving_seg, fixed_seg = moving_batch["seg"], fixed_batch["seg"]  # (B, 1, ...)
        if self.multi_head:
            moving_seg_list, fixed_seg_list = self.separate_seg(moving_seg), self.separate_seg(fixed_seg)  # N x (B, 1, ...)
            loss_organ_list = self.args.organ_list
        else:
            moving_seg_list, fixed_seg_list = [moving_seg], [fixed_seg]  # 1 x (B, 1, ...)
            loss_organ_list = ["all"]
        warped_seg_list = [
            self.warp(ms, ddf, one_hot_moving=self.training or semi_supervision)
            for ms, ddf in zip(moving_seg_list, ddf_list)
        ]  # num_class x (B, 9, ...) or num_class x (B, 1, ...)
        if self.training:
            return self.get_loss(warped_seg_list, fixed_seg_list, ddf_list, loss_organ_list)
        else:
            if semi_supervision and semi_mode == "eval":
                assert not self.multi_head, "semi-supervision does not support multi-head"
                warped_seg = warped_seg_list[0]  # (B, 9, ...)
                warped_t2w = self.warp(moving_batch["t2w"], ddf_list[0], one_hot_moving=False, t2w=True)
                return {
                    "seg": warped_seg,
                    "t2w": warped_t2w
                }
            else:
                warped_seg = warped_seg_list[0]
                for ws in warped_seg_list[1:]:
                    warped_seg += ws * (warped_seg == 0)
                binary = {"seg": warped_seg}
                if not self.multi_head:
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


class NewLocalNet(RegUNet):
    """
    Reimplementation of LocalNet, based on:
    `Weakly-supervised convolutional neural networks for multimodal image registration
    <https://doi.org/10.1016/j.media.2018.07.002>`_.
    `Label-driven weakly-supervised learning for multimodal deformable image registration
    <https://arxiv.org/abs/1711.01666>`_.

    Adapted from:
        DeepReg (https://github.com/DeepRegNet/DeepReg)
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_channel_initial: int,
        extract_levels: Tuple[int],
        out_kernel_initializer: Optional[str] = "kaiming_uniform",
        out_activation: Optional[str] = None,
        out_channels: int = 3,
        pooling: bool = True,
        use_addictive_sampling: bool = True,
        concat_skip: bool = False,
    ):
        """
        Args:
            spatial_dims: number of spatial dims
            in_channels: number of input channels
            num_channel_initial: number of initial channels
            out_kernel_initializer: kernel initializer for the last layer
            out_activation: activation at the last layer
            out_channels: number of channels for the output
            extract_levels: list, which levels from net to extract. The maximum level must equal to ``depth``
            pooling: for down-sampling, use non-parameterized pooling if true, otherwise use conv3d
            use_addictive_sampling: whether use additive up-sampling layer for decoding.
            concat_skip: when up-sampling, concatenate skipped tensor if true, otherwise use addition
        """
        self.use_additive_upsampling = use_addictive_sampling
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_channel_initial=num_channel_initial,
            extract_levels=extract_levels,
            depth=max(extract_levels),
            out_kernel_initializer=out_kernel_initializer,
            out_activation=out_activation,
            out_channels=out_channels,
            pooling=pooling,
            concat_skip=concat_skip,
            encode_kernel_sizes=[7] + [3] * max(extract_levels),
        )

    def build_bottom_block(self, in_channels: int, out_channels: int):
        kernel_size = self.encode_kernel_sizes[self.depth]
        return get_conv_block(
            spatial_dims=self.spatial_dims, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )

    def build_up_sampling_block(self, in_channels: int, out_channels: int) -> nn.Module:
        if self.use_additive_upsampling:
            return AdditiveUpSampleBlock(
                spatial_dims=self.spatial_dims, in_channels=in_channels, out_channels=out_channels
            )

        return get_deconv_block(spatial_dims=self.spatial_dims, in_channels=in_channels, out_channels=out_channels)


class AdditiveUpSampleBlock(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int):
        super().__init__()
        self.deconv = get_deconv_block(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = [size * 2 for size in x.shape[2:]]
        deconved = self.deconv(x)
        resized = F.interpolate(x, output_size)
        resized = torch.sum(torch.stack(resized.split(split_size=resized.shape[1] // 2, dim=1), dim=-1), dim=-1)
        out: torch.Tensor = deconved + resized
        return out


class ConsistencyLoss(nn.Module):
    def __init__(self):
        super(ConsistencyLoss, self).__init__()
        self.warp = Warp()
        self.loss_fn = MSELoss()

    def forward(self, s_ddf, t_ddf, affine_ddf):
        """
        :param s_ddf: (B, 3, W, H, D)
        :param t_ddf: (B, 3, W, H, D)
        :param affine_ddf: (B, 3, W, H, D)
        :return:
        """
        s_ddf = affine_ddf + self.warp(s_ddf, affine_ddf)
        return self.loss_fn(s_ddf, t_ddf)