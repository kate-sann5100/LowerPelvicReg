import numpy as np
from monai.transforms import RandSpatialCropd

from torch.utils.data import Dataset

from data.dataset_utils import get_institution_patient_dict, get_transform, sample_pair, get_img


class RegDataset(Dataset):

    def __init__(self, args, mode):
        super(RegDataset, self).__init__()
        self.args = args
        self.mode = mode

        self.seg_path, self.image_path = f"{args.data_path}/data", f"{args.data_path}/data"

        institution_patient_dict = get_institution_patient_dict(
            data_path=args.data_path,
            mode=mode,
        )

        self.img_list = []
        for ins, patient_list in institution_patient_dict.items():
            self.img_list.extend([(p, ins) for p in patient_list])
        if self.mode != "train":
            self.val_pair = []
            # for each query image
            for moving_p, moving_ins in self.img_list:
                # for each institution
                for fixed_ins, patient_list in institution_patient_dict.items():
                    while True:
                        fixed_p = patient_list[np.random.randint(0, len(patient_list))]
                        if fixed_p != moving_p:
                            break
                    self.val_pair.append([(moving_p, moving_ins), (fixed_p, fixed_ins)])

        self.transform = get_transform(
            augmentation=self.mode == "train",
            size=[args.size[0], args.size[1], 76],
            resolution=args.resolution
        )

        self.random_crop = RandSpatialCropd(
            keys=["t2w", "mask", "seg"],
            roi_size=(args.size[0], args.size[1], 10),
            random_size=False
        )

    def __len__(self):
        return len(self.img_list) if self.mode == "train" else len(self.val_pair)

    def __getitem__(self, idx):
        if self.mode == "train":
            moving = idx
            fixed = sample_pair(idx, len(self.img_list))
            moving, fixed = self.img_list[moving], self.img_list[fixed]
        else:
            moving, fixed = self.val_pair[idx]

        moving = get_img(moving, self.transform, self.image_path, self.seg_path, self.args)
        fixed = get_img(fixed, self.transform, self.image_path, self.seg_path, self.args)

        return moving, fixed
