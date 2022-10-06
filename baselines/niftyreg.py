import os
from shutil import copy

import torch
import numpy as np
import nibabel as nib
from torch.backends import cudnn

from torch.utils.data import DataLoader

from data.dataset import RegDataset
from data.dataset_utils import get_transform
from utils.meter import DiceMeter, HausdorffMeter
from utils.train_eval_utils import get_parser, set_seed



def main():
    args = get_parser()
    cudnn.benchmark = False
    cudnn.deterministic = True
    set_seed(args.manual_seed)
    evaluate(args)


def evaluate(args):
    save_dir = "/niftyreg_vis" if args.vis else None
    if save_dir is not None and not os.path.exists(save_dir):
        os.mkdir(save_dir)

    nifty_reg = NiftyReg(args, save_dir)
    val_dataset = RegDataset(args=args, mode="test")
    val_loader = DataLoader(val_dataset, batch_size=1)

    # dice_meter = DiceMeter(writer=None, test=True)
    # hausdorff_meter = HausdorffMeter(writer=None, test=True)

    for step, (moving, fixed) in enumerate(val_loader):
        nifty_reg.register(moving, fixed)
        # dice_meter.update(
        #     binary["seg"], fixed["seg"],
        #     name=moving[0],
        #     fixed_ins=fixed[1]
        # )


class NiftyReg:
    def __init__(self, args, save_dir):
        temp_dir = "/temp"
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        self.temp_dir = temp_dir

        self.moving_img_path = f"{self.temp_dir}/moving_img.nii"
        self.fixed_img_path = f"{self.temp_dir}/fixed_img.nii"
        self.moving_seg_path = f"{self.temp_dir}/moving_mask.nii"
        self.warped_img_path = f"{self.temp_dir}/warped_img.nii"
        self.warped_seg_path = f"{self.temp_dir}/warped_mask.nii"
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
            self.moving_seg_path: moving_batch["mask"],  # (1, 1, ...)
            self.fixed_img_path: fixed_batch["t2w"]  # (1, 1, ...)
        }
        for k, v in tensor2nii_dict.items():
            nii = nib.Nifti1Image(
                v[0].reshape(*self.args.sz).detach().cpu().numpy().astype(dtype=np.float32),
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
            f"-flo {self.moving_img_path} "
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
            save_warped_seg_path = f"{self.save_dir}/{moving_name}_{fixed_name}_mask.nii"
            save_ddf_path = f"{self.save_dir}/{moving_name}_{fixed_name}_ddf.nii"
            copy(self.warped_img_path, save_warped_img_path)
            copy(self.warped_seg_path, save_warped_seg_path)
            copy(self.ddf_path, save_ddf_path)

    def nii2tensor(self):
        transform = get_transform(False, self.args.size, self.args.resolution)
        warped_result = transform({
            "t2w": self.warped_img_path,
            "seg": self.warped_seg_path,
        })
        nib_seg = nib.load(self.warped_seg_path)
        nib_seg = nib_seg.get_fdata()
        nib_seg = torch.from_numpy(nib_seg[None, ...])
        print(torch.equal(nib_seg, warped_result["seg"]))
        exit()

    def register(self, moving_batch, fixed_batch):
        self.tensor2nii(moving_batch, fixed_batch)
        self.f3d(moving_batch, fixed_batch)
        self.nii2tensor()


if __name__ == '__main__':
    print(1)
    main()