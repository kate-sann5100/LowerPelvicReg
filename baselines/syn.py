import os

import ants
import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader
from monai.transforms import Spacingd
from torch.backends import cudnn

from data.dataset import RegDataset
from utils.meter import DiceMeter, HausdorffMeter
from utils.train_eval_utils import get_parser, set_seed, save_result_dicts


def main():
    args = get_parser()
    set_seed(args.manual_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    save_dir = "niftyreg_result"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    syn = Syn(args, save_dir)
    val_dataset = RegDataset(args=args, mode="val")
    val_loader = DataLoader(val_dataset, batch_size=1)
    dice_meter = DiceMeter(writer=None, test=True)
    hausdorff_meter = HausdorffMeter(writer=None, test=True)
    for step, (moving, fixed) in enumerate(val_loader):
        reg_result = syn.register(moving, fixed)
        dice_meter.update(
            pred_binary=reg_result["seg"],
            seg=fixed["seg"],
            name=moving["name"],
            fixed_ins=fixed["ins"]
        )

        spacingd = Spacingd(["pred", "gt"], pixdim=[1, 1, 1], mode="nearest")
        meta_data = {"affine": moving["t2w_meta_dict"]["affine"][0]}
        resampled = spacingd(
            {
                "pred": reg_result["seg"][0],
                "gt": moving["seg"][0],
                "pred_meta_dict": meta_data,
                "gt_meta_dict": meta_data.copy()
            }
        )
        hausdorff_meter.update(
            resampled["pred"].unsqueeze(0), resampled["gt"].unsqueeze(0),
            name=moving["name"], fixed_ins=fixed["ins"]
        )
        break

    dice_metric, dice_result_dict = dice_meter.get_average(step=None)
    hausdorff_metric, hausdorff_result_dict = hausdorff_meter.get_average(step=None)
    save_result_dicts(save_dir, dice_result_dict, hausdorff_result_dict)


class Syn:
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    @staticmethod
    def load_cache(file_name):
        proxy = nib.load(file_name)
        data = proxy.get_fdata()
        proxy.uncache()
        return data

    def register(self, moving_batch, fixed_batch):
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
        moving_img = moving_batch["t2w"].squeeze(0).squeeze(0).numpy()
        moving_seg = moving_batch["seg"].squeeze(0).squeeze(0).numpy()
        fixed_img = fixed_batch["t2w"].squeeze(0).squeeze(0).numpy()
        fixed_seg = fixed_batch["seg"].squeeze(0).squeeze(0).numpy()

        moving_img = ants.from_numpy(moving_img)
        moving_seg = ants.from_numpy(moving_seg.astype(np.float32))
        fixed_img = ants.from_numpy(fixed_img)
        fixed_seg = ants.from_numpy(fixed_seg.astype(np.float32))

        reg_result = ants.registration(
            fixed=fixed_img, moving=moving_img,
            type_of_transform="SyNOnly",
            reg_iterations=(160, 80, 40),
            syn_metric="meansquares"
        )
        warped_img = reg_result["warpedmovout"].numpy()
        warped_seg = ants.apply_transforms(
            fixed=fixed_seg,
            moving=moving_seg,
            transformlist=reg_result["fwdtransforms"],
            interpolator="nearestNeighbor"
        ).numpy()
        ddf = np.array(
            self.load_cache(reg_result["fwdtransforms"][0]),
            dtype="float32",
            order="C"
        )  # (..., 1, 3)

        result_dict = {
            "t2w": torch.from_numpy(warped_img[None, None, ...]),
            "seg": torch.from_numpy(warped_seg[None, None, ...]),
            "ddf": ddf
        }
        return result_dict


if __name__ == '__main__':
    main()