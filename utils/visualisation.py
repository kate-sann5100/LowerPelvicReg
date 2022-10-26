import os
import nibabel as nib
import numpy as np


class Visualisation:

    def __init__(self, save_path):
        self.save_path = save_path
        print(save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    def vis(self, moving, fixed, pred):
        affine = np.array([[0.75, 0, 0, 0], [0, 0.75, 0, 0], [0, 0, 2.5, 0], [0, 0, 0, 1]])
        moving_name = moving.pop("name")[0]
        fixed_name = fixed.pop("name")[0]
        description = f"{moving_name}_{fixed_name}"
        sz = moving["t2w"].shape

        # warped_t2w, warped_seg

        vis_dict = {
            "warped_t2w": pred["t2w"],
            "warped_seg": pred["seg"],
        }

        for k, v in vis_dict.items():
            img = nib.Nifti1Image(
                v[0].reshape(*sz[-3:]).detach().cpu().numpy().astype(dtype=np.float32),
                affine=affine
            )
            nib.save(img, f"{self.save_path}/{description}_{k}.nii")
