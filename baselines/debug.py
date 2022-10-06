import os
from shutil import copy

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import RegDataset
from utils import config
from utils.meter import DiceMeter


def main():
    args = config.load_cfg_from_cfg_file("config/local.yaml")
    args.vis = False

    save_dir = "niftyreg_vis" if args.vis else None
    if save_dir is not None and not os.path.exists(save_dir):
        os.mkdir(save_dir)
    nifty_reg = NiftyReg(args, save_dir)
    val_dataset = RegDataset(args=args, mode="val")
    val_loader = DataLoader(val_dataset, batch_size=1)
    dice_meter = DiceMeter(writer=None, test=True)
    for step, (moving, fixed) in enumerate(val_loader):
        reg_result = nifty_reg.register(moving, fixed)
        dice_meter.update(
            pred_binary=reg_result["seg"],
            seg=fixed["seg"],
            name=moving["name"],
            fixed_ins=fixed["ins"]
        )
        break
    dice_metric, dice_result_dict = dice_meter.get_average(step=None)
    print(dice_metric)
    print(dice_result_dict)


class NiftyReg:
    def __init__(self, args, save_dir):
        temp_dir = "nifty_temp"
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        self.temp_dir = temp_dir

        self.moving_img_path = f"{self.temp_dir}/moving_img.nii"
        self.fixed_img_path = f"{self.temp_dir}/fixed_img.nii"
        self.moving_seg_path = f"{self.temp_dir}/moving_seg.nii"
        self.warped_img_path = f"{self.temp_dir}/warped_img.nii"
        self.warped_seg_path = f"{self.temp_dir}/warped_seg.nii"
        self.cpp_path = f"{self.temp_dir}/cpp.nii"
        self.def_path = f"{self.temp_dir}/dis.nii"
        self.ddf_path = f"{self.temp_dir}/ddf.nii"

        self.args = args
        self.save_dir = save_dir
        self.vis = args.vis
        self.niftyreg_path = "/Users/yiwenli/Projects/niftyreg/reg-apps"  # path to niftyreg installation

    def tensor2nii(self, moving_batch, fixed_batch):
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

        affine = np.array([[0.75, 0, 0, 0], [0, 0.75, 0, 0], [0, 0, 2.5, 0], [0, 0, 0, 1]])
        tensor2nii_dict = {
            self.moving_img_path: moving_batch["t2w"],  # (1, 1, ...)
            self.moving_seg_path: moving_batch["seg"],  # (1, 1, ...)
            self.fixed_img_path: fixed_batch["t2w"]  # (1, 1, ...)
        }
        for k, v in tensor2nii_dict.items():
            nii = nib.Nifti1Image(
                v[0].reshape(*self.args.size).detach().cpu().numpy().astype(dtype=np.float32),
                affine=affine
            )
            nib.save(nii, k)

    def f3d(self, moving_batch, fixed_batch):
        os.system(
            f"{self.niftyreg_path}/reg_f3d "
            f"-be 0.0002 "
            f"--ssd "
            f"-ref {self.fixed_img_path} "
            f"-flo {self.moving_img_path} "
            f"-res {self.warped_img_path} "
            f"-cpp {self.cpp_path}"
        )
        os.system(
            f"{self.niftyreg_path}/reg_resample "
            f"-ref {self.fixed_img_path} "
            f"-flo {self.moving_seg_path} "
            f"-res {self.warped_seg_path} "
            f"-cpp {self.cpp_path} "
            f"-inter 0"
        )
        os.system(
            f"{self.niftyreg_path}/reg_transform "
            f"-ref {self.fixed_img_path} "
            f"-cpp2def {self.cpp_path} {self.def_path}"
        )
        os.system(
            f"{self.niftyreg_path}/reg_transform "
            f"-ref {self.fixed_img_path} "
            f"-def2disp {self.def_path} {self.ddf_path}"
        )

        if self.vis:
            moving_name, fixed_name = moving_batch["name"], fixed_batch["name"]
            save_warped_img_path = f"{self.save_dir}/{moving_name}_{fixed_name}_img.nii"
            save_warped_seg_path = f"{self.save_dir}/{moving_name}_{fixed_name}_seg.nii"
            save_ddf_path = f"{self.save_dir}/{moving_name}_{fixed_name}_ddf.nii"
            copy(self.warped_img_path, save_warped_img_path)
            copy(self.warped_seg_path, save_warped_seg_path)
            copy(self.ddf_path, save_ddf_path)

    @staticmethod
    def nii2tensor(nii_path):
        x = nib.load(nii_path)
        x = x.get_fdata()
        x = torch.from_numpy(x[None, None, ...])
        return x

    def register(self, moving_batch, fixed_batch):
        self.tensor2nii(moving_batch, fixed_batch)
        self.f3d(moving_batch, fixed_batch)
        return {
            "t2w": self.nii2tensor(self.warped_img_path),
            "seg": self.nii2tensor(self.warped_seg_path),
            "ddf": self.nii2tensor(self.ddf_path)
        }


if __name__ == '__main__':
    main()