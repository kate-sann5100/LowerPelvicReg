import os
import nibabel as nib
import numpy as np


class Visualisation:

    def __init__(self, save_path):
        self.save_path = save_path
        print(f"visualisation will be saved in {save_path}")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    def vis(self, moving, fixed, pred=None, prefix=None):
        affine = np.array([[0.75, 0, 0, 0], [0, 0.75, 0, 0], [0, 0, 2.5, 0], [0, 0, 0, 1]])
        moving_name = moving.pop("name")
        fixed_name = fixed.pop("name")
        description = [f"{mn}_{fn}" for mn, fn in zip(moving_name, fixed_name)]
        if prefix is not None:
            description = [f"{d}_{prefix}" for d in description]
        sz = moving["t2w"].shape

        # warped_t2w, warped_seg

        vis_dict = {
            "moving_t2w": moving["t2w"],
            "fixed_t2w": fixed["t2w"],
        }
        if "seg" in moving.keys():
            vis_dict["moving_seg"] = moving["seg"]
            vis_dict["fixed_seg"] = fixed["seg"]
        if "cut_mask" in moving.keys():
            vis_dict["moving_cutmask"] = moving["cut_mask"]
        if pred is not None:
            vis_dict["warped_t2w"] = pred["t2w"]
            vis_dict["warped_seg"] = pred["seg"]

        for k, v in vis_dict.items():
            for i, d in enumerate(description):
                img = nib.Nifti1Image(
                    v[i].reshape(*sz[-3:]).detach().cpu().numpy().astype(dtype=np.float32),
                    affine=affine
                )
                nib.save(img, f"{self.save_path}/{d}_{k}.nii")
