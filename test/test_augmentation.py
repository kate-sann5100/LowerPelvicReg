import numpy
import torch
from monai.networks.blocks import Warp

from data.strong_aug import RandAffine


def test_augmentation(size):
    """
    randomly generate a moving and fixed pair with known ddf
    augment fixed, and the ddf accordingly
    assert the augmented ddf deform moving into augmented fixed
    :param size:
    :return:
    """
    rand_affine = RandAffine(spatial_size=size)
    warp = Warp(mode="bilinear", padding_mode="zeros")
    moving = torch.rand(1, *size)  # (1, ...)
    ddf = torch.rand(len(size), *size)  # (nun_dim, ...)
    fixed = warp(
        moving.unsqueeze(0), ddf.unsqueeze(0)
    ).squeeze(0)  # (1, ...)

    augmented_fix = {"t2w": fixed}
    rand_affine(augmented_fix)  # (1, ...)
    moving, fixed = moving.unsqueeze(0), fixed.unsqueeze(0)
    augmented_fix = {k: v.unsqueeze(0) for k, v in augmented_fix.items()}
    augmented_ddf = augmented_fix["affine_ddf"] + warp(ddf.unsqueeze(0), augmented_fix["affine_ddf"])
    warped_augmented_fix = warp(moving, augmented_ddf)
    assert torch.equal(warped_augmented_fix, augmented_fix["t2w"])


if __name__ == '__main__':
    test_augmentation((10, 10, 10))