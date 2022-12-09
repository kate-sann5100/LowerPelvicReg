import numpy as np
import torch

import matplotlib.pyplot as plt
from monai.networks.blocks import Warp
from data.strong_aug import RandAffine


def generate_moving(size=(100, 100)):
    mesh_points = [torch.arange(0, dim) for dim in size]
    mesh_points = torch.meshgrid(*mesh_points)
    img = torch.sin(mesh_points[0] * 0.05) * torch.cos(mesh_points[1] * 0.052)
    return img


def generate_pair(size=None, moving=None, affine=False):
    """
    randomly generate a moving and fixed pair with known ddf
    :param size:
    :return:
    moving: (W, H)
    fixed: (W, H)
    ddf: (2, W, H)
    """
    warp = Warp(mode="bilinear", padding_mode="zeros")
    if moving is None:
        moving = torch.rand(size)  # (...)
    else:
        size = moving.shape
    if affine:
        fixed_dict = augment(moving)
        fixed = fixed_dict["t2w"][0]
        ddf = fixed_dict["affine_ddf"]
    else:
        ddf = torch.rand(len(size), *size) * size[0]  # (nun_dim, ...)
        fixed = warp(
            moving[None, None, ...], ddf.unsqueeze[None, ...]
        )  # (...)
    return moving, fixed, ddf


def augment(img):
    """
    augment img
    :param img: (W, H)
    :return:
    augmented_img: dict with keys
    - t2w: (1, W, H)
    - affine_ddf: (2, W, H)
    """
    rand_affine = RandAffine(spatial_size=img.shape)
    augmented_img = {"t2w": img.unsqueeze(0)}  # (1, ...)
    rand_affine(augmented_img)  # (1, ...)
    assert not torch.equal(img, augmented_img["t2w"][0]), "no augmentation applied"
    return augmented_img


def compare(moving, augmented_fix, ddf):
    """
    assert the augmented ddf deform moving into augmented fixed
    :param moving: (W, H)
    :param augmented_fix: dict with keys
    - t2w: (1, W, H)
    - affine_ddf: (2, W, H)
    :param ddf: (2, W, H)
    :return:
    result: bool
    augmented_warped: (W, H)
    """
    warp = Warp(mode="bilinear", padding_mode="zeros")

    moving = moving[None, None, ...]  # (1, 1, W, H)
    augmented_fix = {k: v[None, ...] for k, v in augmented_fix.items()}  # (1, C, W, H)
    ddf = ddf[None, ...]  # (1, 2, W, H)
    augmented_ddf = augmented_fix["affine_ddf"] + warp(ddf, augmented_fix["affine_ddf"])
    augmented_warped = warp(moving, augmented_ddf)  # (1, C, W, H)
    if not torch.equal(augmented_warped, augmented_fix["t2w"]):
        # print(warped_augmented_fix - augmented_fix["t2w"])
        return False, augmented_warped[0, 0]
    return True, augmented_warped[0, 0]


def vis():
    moving = generate_moving(size=(100, 100))  # (W, H)
    moving, fixed, ddf = generate_pair(moving=moving, affine=True)  # (W, H), (W, H), (2, W, H)

    augmented_fixed = augment(fixed)  # (C, W, H)
    result, augmented_warped = compare(moving, augmented_fixed, ddf)  # (W, H)
    print(result)
    axs = plt.figure(constrained_layout=True).subplots(2, 2, sharex=True, sharey=True)
    vis_dict = {
        "moving": (axs[0, 0], moving),
        "fixed": (axs[0, 1], fixed),
        "augmented_warped": (axs[1, 0], augmented_warped),
        "augmented_fix": (axs[1, 1], augmented_fixed["t2w"].reshape(100, 100)),
    }
    for title, (ax, img) in vis_dict.items():
        ax.set(title=title, aspect=1, xticks=[], yticks=[])
        ax.matshow(np.array(img))
    plt.show()


# def test_augmentation(size):
#     failed_pair_count = 0
#     for pair_id in range(10):
#         moving, fixed, ddf = generate_pair(size)
#         failed_augment_count = 0
#         for _ in range(10):
#             result, _, _ = augment(moving, fixed, ddf)
#             if not result:
#                 failed_augment_count += 1
#         if failed_augment_count > 0:
#             print(f"pair{pair_id}: failed {failed_augment_count}/10")
#     print(f"failed {failed_pair_count}/10")


if __name__ == '__main__':
    for _ in range(10):
        vis()