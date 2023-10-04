import numpy as np
from monai.transforms import RandSpatialCropd

from torch.utils.data import Dataset

from data.dataset_utils import get_institution_patient_dict, get_transform, sample_pair, get_img, sample_patch, \
    get_patch
from data.strong_aug import RandAffine, Cut


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


class PatchDataset(Dataset):

    def __init__(self, args, mode):
        super(PatchDataset, self).__init__()
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
            self.val_pair = [
                (img, sample_patch(self.args), sample_patch(self.args))
                for img in self.img_list
            ]

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
            img = self.img_list[idx]
            moving, fixed = sample_patch(self.args), sample_patch(self.args)
        else:
            img, moving, fixed = self.val_pair[idx]

        img = get_img(img, self.transform, self.image_path, self.seg_path, self.args)
        moving, fixed = get_patch(img, moving, self.args), get_patch(img, fixed, self.args)

        return moving, fixed


class SemiDataset(Dataset):

    def __init__(self, args, mode, label):
        super(SemiDataset, self).__init__()
        self.args = args
        self.mode = mode
        self.label = label

        self.seg_path, self.image_path = f"{args.data_path}/data", f"{args.data_path}/data"

        institution_patient_dict = get_institution_patient_dict(
            data_path=args.data_path,
            mode=mode,
        )

        self.img_list = []
        for ins, patient_list in institution_patient_dict.items():
            if mode == "train":
                label_num = int(len(patient_list) * args.label_ratio)
                patient_list = patient_list[:label_num] if label else patient_list[label_num:]
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

        self.rand_affine = RandAffine(
            aug_multiplier=args.aug_multiplier,
            spatial_size=args.size,
        )
        self.cut = Cut(args)

    def __len__(self):
        return len(self.img_list) if self.mode == "train" else len(self.val_pair)

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        moving with keys:
        -"t2w": (1, W, H, D)
        -"seg": (1, W, H, D) or key not exist
        -"ins": int
        -"name": str
        fixed with keys:
        -"t2w": (1, W, H, D)
        -"seg": (1, W, H, D) or key not exist
        -"affine_ddf": (3, W, H, D)
        -"ins": int
        -"name": str
        aug_moving with keys:
        -"t2w": (1, W, H, D)
        -"seg": (1, W, H, D) or key not exist
        -"ins": int
        -"name": str
        -"cut_mask": (1, W, H, D)
        aug_fixed with keys:
        -"t2w": (1, W, H, D)
        -"seg": (1, W, H, D) or key not exist
        -"ins": int
        -"name": str
        -"affine_ddf": (3, W, H, D)
        """
        if self.mode == "train":
            moving = idx
            fixed = sample_pair(idx, len(self.img_list))
            moving, fixed = self.img_list[moving], self.img_list[fixed]
        else:
            moving, fixed = self.val_pair[idx]

        moving = get_img(moving, self.transform, self.image_path, self.seg_path, self.args)
        fixed = get_img(fixed, self.transform, self.image_path, self.seg_path, self.args)
        if self.label:
            return moving, fixed
        else:
            del moving["seg"]
            del fixed["seg"]
            aug_fixed = self.rand_affine(fixed)
            aug_moving = self.cut(moving, aug_fixed)
            return moving, fixed, aug_moving, aug_fixed


class AtlasDataset(Dataset):

    def __init__(self, args):
        super(AtlasDataset, self).__init__()
        self.args = args
        self.seg_path, self.image_path = f"{args.data_path}/data", f"{args.data_path}/data"

        institution_patient_dict = get_institution_patient_dict(
            data_path=args.data_path,
            mode=mode,
        )

        self.img_list = []
        for ins, patient_list in institution_patient_dict.items():
            if mode == "train":
                label_num = int(len(patient_list) * args.label_ratio)
                patient_list = patient_list[:label_num] if label else patient_list[label_num:]
            self.img_list.extend([(p, ins) for p in patient_list])

        self.transform = get_transform(
            augmentation=False,
            size=[args.size[0], args.size[1], 76],
            resolution=args.resolution
        )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        img with keys:
        -"t2w": (1, W, H, D)
        -"seg": (1, W, H, D)
        -"ins": int
        -"name": str
        """
        img = self.img_list[idx]
        img = get_img(img, self.transform, self.image_path, self.seg_path, self.args)
        return img
