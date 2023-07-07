import torch
from monai.data import DataLoader
from monai.networks.blocks import Warp

from data.dataset import SemiDataset
from data.strong_aug import RandAffine, Cut
from utils.train_eval_utils import get_parser


def main():
    """
    test the transformation of ddf led by augmentations
    - case 1: fixed = moving so ddf is all zero, assert warp(aug_moving, aug_ddf) = aug_fixed
              visualisation
    - case 2: no augmentation applied, so aug_ddf = ddf
              assert
    :return:
    """
    args = get_parser()
    ul_dataset = SemiDataset(args=args, mode="train", label=False)
    ul_loader = DataLoader(
        ul_dataset,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        persistent_workers=False,
    )
    print(f"unlabelled dataset of size {len(ul_loader)}")

    for i, (ul_moving, ul_fixed, aug_moving, aug_fixed) in enumerate(ul_loader):
        ddf = torch.zeros(size=(1, 3, *args.size))
        # print(f"sampled ddf, sampled aug")
        # test_aug(moving_batch=ul_moving,
        #          ddf=ddf,
        #          aug_multiplier=1.0, cut_ratio=(0.5, 0.5), args=args)
        print(f"sampled ddf, zero aug")
        test_aug(moving_batch=ul_moving,
                 ddf=ddf,
                 aug_multiplier=0, cut_ratio=(0, 0), args=args)
        print(f"zero ddf, sampled aug")
        ddf = torch.zeros(size=(1, 3, *args.size))
        test_aug(moving_batch=ul_moving,
                 ddf=ddf,
                 aug_multiplier=1.0, cut_ratio=(0.5, 0.5), args=args)


def test_aug(moving_batch, ddf, aug_multiplier, cut_ratio, args):
    """
    :param moving_batch:
        -"t2w": (B, 1, W, H, D)
        -"ins": int
        -"name": str
    :param ddf: (B, 3, W, H, D)
    :param aug_multiplier: how big the augmentation is, when set to zero, no affine augmentation
    :param cut_ratio: tuple of (low, high)
    :return:
    """
    # generate fixed
    fixed_batch = {
        "t2w": Warp()(moving_batch["t2w"], ddf),
        "ins": moving_batch["ins"],
        "name": moving_batch["name"]
    }
    # augmentation
    args.aug_multiplier = aug_multiplier
    args.cut_ratio = cut_ratio
    rand_affine = RandAffine(aug_multiplier=args.aug_multiplier)
    cut = Cut(args)
    moving, fixed = moving_batch.copy(), fixed_batch.copy()
    moving["t2w"] = moving_batch["t2w"][0]
    fixed["t2w"] = fixed_batch["t2w"][0]
    aug_fixed = rand_affine(fixed)
    aug_moving = cut(moving, fixed)
    aug_moving_batch, aug_fixed_batch = aug_moving.copy(), aug_fixed.copy()
    aug_moving_batch["t2w"] = aug_moving["t2w"].unsqueeze(0)  # (B, 1, W, H, D)
    aug_moving_batch["cut_mask"] = aug_moving["cut_mask"].unsqueeze(0)  # (B, 1, W, H, D)
    aug_fixed_batch["t2w"] = aug_fixed["t2w"].unsqueeze(0)  # (B, 1, W, H, D)
    aug_fixed_batch["affine_ddf"] = aug_fixed["affine_ddf"].unsqueeze(0)  # (B, 3, W, H, D)
    print("augmented")
    # transform ddf
    affine_ddf = aug_fixed_batch["affine_ddf"]  # (B, 3, W, H, D)
    cut_mask = aug_moving_batch["cut_mask"]  # (B, 1, W, H, D)
    aug_ddf = affine_ddf.to(ddf.cuda()) + Warp().cuda()(ddf.cuda(), affine_ddf.to(ddf))  # (B, 3, W, H, D)
    aug_ddf = (1 - cut_mask.to(ddf)) * aug_ddf  # (B, 3, W, H, D)
    print("ddf transformed")
    # warp augmented pair
    aug_warped_t2w = Warp()(aug_moving_batch["t2w"], aug_ddf)  # (B, 1, W, H, D)
    error = torch.unique((aug_warped_t2w - aug_fixed_batch["t2w"]) / aug_fixed_batch["t2w"])
    print(error)
    # max_error = max(error)
    # print(max_error)
    assert torch.equal(aug_warped_t2w, aug_fixed_batch["t2w"])


if __name__ == '__main__':
    main()