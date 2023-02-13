import numpy as np
import torch
from torch.nn import functional as F

from monai.transforms import (
    AddChanneld,
    CenterSpatialCropd,
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    ScaleIntensityd,
    Spacingd,
    SpatialPadd,
    ToTensord,
    RandAffined
)


organ_list = ["BladderMask", "BoneMask", "ObdInternMask", "TZ",
              "CG", "RectumMask", "SV", "NVB"]
organ_index_dict = {organ: i + 1 for i, organ in enumerate(organ_list)}


def get_institution_patient_dict(data_path, mode):
    """
    choose images based on institution
    :param data_path: str
    :param mode: train/val/test
    :return: dict
    """

    # divide images by institution
    institution_patient_dict = {i: [] for i in range(1, 8)}
    with open(f'{data_path}/institution.txt') as f:
        patient_ins_list = f.readlines()
    for patient_ins in patient_ins_list:
        patient, ins = patient_ins[:-1].split(" ")
        institution_patient_dict[int(ins)].append(patient)

    for k, v in institution_patient_dict.items():
        if mode == "train":
            institution_patient_dict[k] = v[:-len(v)//4]
        if mode == "val":
            institution_patient_dict[k] = v[:-len(v)//4-2]
        else:
            institution_patient_dict[k] = v[-len(v)//4-2:]

    return institution_patient_dict


def get_transform(augmentation, size, resolution):
    pre_augmentation = [
        LoadImaged(keys=["t2w", "seg"]),
        AddChanneld(keys=["t2w", "seg"]),
        Spacingd(
            keys=["t2w", "seg"],
            pixdim=resolution,
            mode=("bilinear", "nearest"),
        ),
    ]

    post_augmentation = [
        NormalizeIntensityd(keys=["t2w"]),
        ScaleIntensityd(keys=["t2w"]),
        ToTensord(keys=["t2w", "seg"])
    ]

    if augmentation:
        middle_transform = [
            RandAffined(
                keys=["t2w", "seg"],
                spatial_size=size,
                prob=1.0,
                rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
                shear_range=None,
                translate_range=(20, 20, 4),
                scale_range=(0.15, 0.15, 0.15),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
                as_tensor_output=False,
                device=torch.device('cpu'),
                allow_missing_keys=False
            )
        ]
    else:
        middle_transform = [
            CenterSpatialCropd(keys=["t2w", "seg"], roi_size=size),
            SpatialPadd(
                keys=["t2w", "seg"],
                spatial_size=size,
                method='symmetric',
                mode='constant',
                allow_missing_keys=False
            )
        ]

    return Compose(pre_augmentation + middle_transform + post_augmentation)


def sample_pair(idx, img_list_len):
    """
    :param idx: int
    :param img_list_len: int
    :return: int
    """
    out = idx
    while out == idx:
        out = np.random.randint(img_list_len)
    return out


def get_img(img, transform, image_path, seg_path, args):
    """
    :param img: tuple (name, ins)
    :param transform:
    :param image_path: str
    :param seg_path: str
    :param args
    :return:
    t2w: (1, ...)
    seg: (1, ...)
    name: str
    """
    img_name, ins = img
    x = transform({
        "t2w": f"{image_path}/{img_name}_img.nii",
        "seg": f"{seg_path}/{img_name}_mask.nii",
        "name": img_name,
        "ins": ins
    })
    target_slice = torch.sum(x["seg"], dim=(0, 1, 2)) != 0
    for k in ["t2w", "seg"]:
        x[k] = x[k][..., target_slice]
        x[k] = F.interpolate(
            x[k].unsqueeze(0).to(torch.float),
            size=args.size,
            mode="trilinear" if k == "t2w" else "nearest"
        ).squeeze(0)
    new_seg = torch.zeros_like(x["seg"])
    for organ in args.organ_list:
        organ_index = organ_index_dict[organ]
        new_seg[x["seg"] == organ_index] = organ_index
    x["seg"] = new_seg
    return x


def get_patch(x, pos, args):
    """
    :param x: dict with t2w:(1, ...) seg:(1, ...) name:str
    :param pos: front-left-top coordinate of the patch
    :param args:
    :return:
    """
    w, h, d = args.patch_size
    x = {
        "t2w": x["t2w"][:, pos[0]:pos[0]+w, pos[1]:pos[1]+h, pos[2]:pos[2]+d],
        "seg": x["seg"][:, pos[0]:pos[0]+w, pos[1]:pos[1]+h, pos[2]:pos[2]+d],
        "name": x["name"],
        "pos": pos
    }
    return x


def sample_patch(args):
    w, h, d = np.array(args.size) - np.array(args.patch_size)
    return np.array(
        [int(np.random.uniform()*w), int(np.random.uniform()*h), int(np.random.uniform()*d)]
    )