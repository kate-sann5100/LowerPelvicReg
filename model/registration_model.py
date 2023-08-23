from functools import partial
from typing import Tuple, Union, List, Optional, Callable

import torch
# from torchvision.models import vision_transformer
import numpy as np
from monai.losses import DiceLoss, BendingEnergyLoss
from monai.networks import one_hot
from monai.networks.blocks import Warp
from monai.networks.blocks.regunet_block import get_conv_block, get_deconv_block
from monai.networks.nets import LocalNet, RegUNet
from torch import nn
from torch.nn import functional as F, MSELoss


class Registration(nn.Module):

    def __init__(self, args):
        super(Registration, self).__init__()
        self.args = args
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

        self.transformer = args.transformer
        self.vit_block = EncoderBlock(
            num_heads=8,
            hidden_dim=32 * (2 ** 3),
            mlp_dim=32 * (2 ** 3),
            dropout=0.0,
            attention_dropout=0.0
        ) if args.transformer else None
        self.conv_block = get_conv_block(
            spatial_dims=3,
            in_channels=32 * (2 ** 3),
            out_channels=32 * (2 ** 3),
            kernel_size=3
        ) if args.transformer else None

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
            print(skip.shape)
            encoded = encode_pool(skip)
            skips.append(skip)
        decoded = self.model.bottom_block(encoded)
        if self.transformer:
            b, c, w, h, d = decoded.shape
            decoded = decoded.reshape(b, c, -1).permute(0, 2, 1)  # (B, W*H*D, C)
            decoded = self.vit_block(decoded).permute(0, 2, 1)  # (B, C, W*H*D)
            decoded = decoded.reshape(b, c, w, h, d)
            decoded = self.conv_block(decoded)

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

        ddf = self.forward_output_block(self.model.output_block, outs, image_size)

        return ddf  # (B, 3, H, W, D)

    @staticmethod
    def warp(moving, ddf, t2w=False):
        """
        :param moving: (B, 1, W, H, D)
        :param ddf: (B, 3, W, H, D)
        :param t2w: if input is t2w, warp with "bilinear"
        :return:
        """
        pred = Warp(mode="bilinear")(
            moving if t2w else one_hot(moving, num_classes=9),
            ddf
        )
        return pred  # (B, 1, ...) if t2w else (B, 9, ...)

    def forward(self, moving_batch, fixed_batch, semi_supervision=False):
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
        :return:
        during training with labelled data
        student returns loss_dict with keys
        - "label": (B)
        - "reg": (B)

        during training with unlabelled data
        both student and teacher return ddf of shape (B, 3, W, H, D)

        during evaluation
        both student and teacher return binary with keys
        - "t2w": (B, 1, W, H, D)
        - "seg": (B, 9, W, H, D)
        - "ddf": (B, 3, W, H, D)
        """
        if self.args.input == "img":
            x = torch.cat([moving_batch["t2w"], fixed_batch["t2w"]], dim=1)
        else:
            x = torch.cat([moving_batch["seg"], fixed_batch["seg"]], dim=1)
        ddf = self.forward_localnet(x)  # (B, 3, H, W, D)

        if semi_supervision:
            return ddf  # (B, 3, H, W, D)

        moving_seg, fixed_seg = moving_batch["seg"], fixed_batch["seg"]  # (B, 1, ...), (B, 1, ...)
        warped_seg = self.warp(moving_seg, ddf, t2w=False)  # (B, 9, ...)
        if self.training:
            return self.get_loss(warped_seg, fixed_seg, ddf)
        else:
            binary = {
                "t2w": self.warp(moving_batch["t2w"], ddf, t2w=True),  # (B, 1, H, W, D)
                "seg": warped_seg,  # (B, 9, H, W, D)
                "ddf": ddf,  # (B, 3, H, W, D)
            }
            return binary

    def get_label_loss(self, warped_seg, fixed_seg):
        """
        :param warped_seg: (B, 9, ...)
        :param fixed_seg: (B, 1, ...)
        :return:
        """
        label_loss = {
            "label": self.label_loss(warped_seg, fixed_seg)
        }
        return label_loss

    def get_reg_loss(self, ddf, label_loss):
        """
        :param ddf: (B, 3, ...)
        :return:
        """
        if self.args.reg:
            return {"reg": 0.1 * BendingEnergyLoss()(ddf)}
        else:
            return {"reg": torch.zeros_like(label_loss["label"])}

    def get_loss(self, warped_seg, fixed_seg, ddf):
        """
        :param warped_seg: (B, 9, W, H, D)
        :param fixed_seg: (B, 1, W, H, D)
        :param ddf: (B, 3, W, H, D)
        :return:
        """
        loss_dict = {}
        label_loss = self.get_label_loss(warped_seg, fixed_seg)  # n x scalar
        loss_dict.update(label_loss)
        reg_loss = self.get_reg_loss(ddf, label_loss)
        loss_dict.update(reg_loss)
        return loss_dict


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = True,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


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

    def forward(self, student_aug_ddf, teacher_ddf, affine_ddf, cut_mask):
        """
        transform teacher-predicted ddf to teacher-predicted augmented ddf
        compute distance between student- and teacher-predicted augmented ddf
        :param student_aug_ddf: (B, 3, W, H, D)
        :param teacher_ddf: (B, 3, W, H, D)
        :param affine_ddf: (B, 3, W, H, D)
        :param cut_mask: (B, 1, W, H, D)
        :return:
        """
        # transform teacher-predicted ddf to teacher-predicted augmented ddf
        teacher_aug_ddf = affine_ddf.to(teacher_ddf) + self.warp(teacher_ddf, affine_ddf.to(teacher_ddf))
        teacher_aug_ddf = (1 - cut_mask.to(teacher_ddf)) * teacher_aug_ddf
        return self.loss_fn(student_aug_ddf, teacher_aug_ddf)