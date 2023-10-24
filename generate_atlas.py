import os

import numpy as np
import torch
import nibabel as nib
from monai.metrics import DiceMetric
from monai.networks import one_hot
from monai.networks.blocks import Warp
from torch.backends import cudnn
from torch.cuda import device_count
from torch.utils.data import DataLoader

from data.dataset import AtlasDataset
from data.dataset_utils import organ_list
from model.registration_model import Registration
from utils.train_eval_utils import cuda_batch, get_parser, set_seed


def main():
    args = get_parser()
    cudnn.benchmark = False
    cudnn.deterministic = True
    set_seed(args.manual_seed)
    save_dir = "atlas/upper_bound"
    vis_path = "atlas/upper_bound/vis"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(vis_path):
        os.mkdir(vis_path)
    model = torch.nn.DataParallel(
        Registration(args).cuda()
    )
    ckpt = torch.load("new_ckpt/img128*128*24_vit_labeledonly1.0/best_ckpt.pth")
    model.load_state_dict(ckpt["student"], strict=True)

    # initialise training dataloaders
    dataset = AtlasDataset(args=args)
    init_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        persistent_workers=False,
    )
    update_dataloader = DataLoader(
        dataset,
        batch_size=device_count() * args.batch_size,
        shuffle=True,
        drop_last=True,
        persistent_workers=False,
    )
    atlas = initialise_atlas(init_dataloader, args)
    print(f"atlas initialised...")
    ckpt = {}
    for iter in range(1):
        print(f"iter{iter}...")
        atlas = update_atlas(
            atlas, update_dataloader, model, device_count() * args.batch_size, len(dataset), save_dir, args
        )
        ckpt[f"iter{iter}"] = atlas
        visualise_atlas(atlas, iter, vis_path)
    torch.save(ckpt, f"{save_dir}/ckpt.pth")


def initialise_atlas(dataloader, args):
    """
    Calculate average segmentation logits,
    choose the sample with the highest dice score compared to the average logits
    :param dataloader: with batch size = 1
    :param args:
    :return:
    """
    seg_sum, seg_count = torch.zeros(1, 9, *args.size), 0
    for step, img in enumerate(dataloader):
        cuda_batch(img)
        seg = one_hot(img["seg"], num_classes=9)  # (1, 9, W, H, D)
        if step == 0:
            seg_sum = seg_sum.to(seg)
        seg_sum += seg
        seg_count += 1
    seg_avg = seg_sum / seg_count  # (1, 9, W, H, D)
    atlas, best_metric = None, 0
    for img in dataloader:
        cuda_batch(img)
        one_hot_seg = one_hot(img["seg"], num_classes=9)  # (1, 9, W, H, D)
        mean_dice = torch.mean(
            DiceMetric(include_background=False, reduction="mean")(one_hot_seg, seg_avg)
        )
        if mean_dice > best_metric:
            atlas = img.copy()
            best_metric = mean_dice
    return atlas


def warp(moving, ddf, t2w=False):
    """
    :param moving: (B, 1, W, H, D)
    :param ddf: (B, 3, W, H, D)
    :param t2w: if input is t2w, warp with "bilinear"
    :return:
    """
    pred = Warp(mode="bilinear")(
        moving if t2w else one_hot(moving, num_classes=9),
        ddf
    )
    return pred  # (B, 1, ...) if t2w else (B, 9, ...)


def update_atlas(atlas, dataloader, model, batch_size, num_samples, save_dir, args):
    """
    :param atlas:
        "t2w": (B, 1, ...)
        "seg": (B, 1, ...)
    :param dataloader:
    :param model:
    :param batch_size:
    :param num_samples:
    :param save_dir:
    :param args:
    :return:
    """
    all_ddf = torch.zeros(num_samples, 3, *args.size)
    all_t2w = torch.zeros(num_samples, 1, *args.size)
    all_seg = torch.zeros(num_samples, 9, *args.size)
    cuda_batch(atlas)
    model.eval()
    with torch.no_grad():
        for step, img in enumerate(dataloader):
            cuda_batch(img)
            batch_atlas = {
                "t2w": atlas["t2w"].expand(len(img["t2w"]), 1, *args.size),
                "seg": atlas["seg"],
            }
            cuda_batch(batch_atlas)
            ddf = model(moving_batch=img, fixed_batch=batch_atlas, semi_supervision=True)  # (B, 3, W, H, D)
            all_ddf[step*batch_size: step*batch_size+len(img["t2w"])] = ddf  # (B, 3, W, H, D)
            binary = model(moving_batch=img, fixed_batch=batch_atlas, semi_supervision=False)
            all_t2w[step*batch_size: step*batch_size+len(img["t2w"])] = binary["t2w"]  # (B, 1, W, H, D)
            all_seg[step*batch_size: step*batch_size+len(img["t2w"])] = binary["seg"]  # (B, 9, W, H, D)

            ddf_variance_log = log_ddf_variance(ddf, img, binary)
            visualise_img(img, binary, vis_path=f"{save_dir}/vis_sample")

    with open(f'{save_dir}/var_log.txt', 'w') as f:
        for n in ddf_variance_log.keys():
            f.write(f"{n} \n")
            for k, v in ddf_variance_log[n].items():
                f.write(f"{k}:{v} \n")
    torch.save(ddf_variance_log, f"{save_dir}/var_log.pth")

    var_ddf, avg_ddf = torch.var_mean(ddf, dim=0, keepdim=True)  # (1, 3, W, H, D)
    var_t2w, avg_t2w = torch.var_mean(all_t2w, dim=0, keepdim=True)   # (1, 3, W, H, D)
    var_seg, avg_seg = torch.var_mean(all_seg, dim=0, keepdim=True)  # (1, 9, W, H, D)
    atlas = {
        "t2w": avg_t2w,
        "seg": avg_seg,
        "var_t2w": var_t2w,
        "var_seg": var_seg,
        "var_ddf": var_ddf,
        "all_ddf": all_ddf
    }
    return atlas


def visualise_atlas(atlas, iteration, vis_path):
    affine = np.array([[0.75, 0, 0, 0], [0, 0.75, 0, 0], [0, 0, 2.5, 0], [0, 0, 0, 1]])
    sz = atlas["t2w"].shape

    img = nib.Nifti1Image(
        atlas["t2w"].reshape(*sz[-3:]).detach().cpu().numpy().astype(dtype=np.float32),
        affine=affine
    )
    nib.save(img, f"{vis_path}/{iteration}_t2w.nii")

    seg_binary = torch.argmax(atlas["seg"], dim=1, keepdim=True)  # (1, 1, W, H, D)
    img = nib.Nifti1Image(
        seg_binary.reshape(*sz[-3:]).detach().cpu().numpy().astype(dtype=np.float32),
        affine=affine
    )
    nib.save(img, f"{vis_path}/{iteration}_seg.nii")

    surface_ddf_var = atlas["var_ddf"].to(seg_binary) * (seg_binary > 0)
    for i, dim in enumerate(["W", "H", "D"]):
        img = nib.Nifti1Image(
            surface_ddf_var[:, i].reshape(*sz[-3:]).detach().cpu().numpy().astype(dtype=np.float32),
            affine=affine
        )
        nib.save(img, f"{vis_path}/{iteration}_ddf_var_{dim}.nii")

    for cls in range(1, 9):
        cls_logit = atlas["seg"][:, cls, ...]  # (B, 1, W, H, D)
        img = nib.Nifti1Image(
            cls_logit.reshape(*sz[-3:]).detach().cpu().numpy().astype(dtype=np.float32),
            affine=affine
        )
        nib.save(img, f"{vis_path}/{iteration}_{organ_list[cls - 1]}.nii")


def log_ddf_variance(ddf, img, binary):
    """
    :param ddf: (B, 3, W, H, D)
    :param img:
        -"t2w": (B, 1, W, H, D)
        -"seg": (B, 1, W, H, D)
        -"ins": int
        -"name": str
    :param binary:
        - "t2w": (B, 1, W, H, D)
        - "seg": (B, 9, W, H, D)
        - "ddf": (B, 3, W, H, D)
    :return:
    """
    var, avg = torch.var_mean(ddf, dim=[2, 3, 4])  # (B, 3)
    result = {n: {"all_var": var[i], "all_avg": avg[i]}
              for i, n in enumerate(img["name"])}
    for cls in range(1, 9):
        mask = (img["seg"] == cls)  # (B, 1, W, H, D)
        masked_ddf = ddf * mask  # (B, 3, W, H, D)
        avg = masked_ddf.sum(dim=(2, 3, 4)) / mask.sum(dim=(2, 3, 4))  # (B, 3)
        var = masked_ddf - avg.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, 3, W, H, D)
        var = torch.sum(var * var * mask, dim=(2, 3, 4)) / mask.sum(dim=(2, 3, 4))  # (B, 3)
        for i, n in enumerate(img["name"]):
            result[n][f"{organ_list[cls-1]}_var"] = var[i]
            result[n][f"{organ_list[cls-1]}_avg"] = avg[i]
    return result


def visualise_img(img, binary, vis_path):
    """
    :param img:
        -"t2w": (B, 1, W, H, D)
        -"seg": (B, 1, W, H, D)
        -"ins": int
        -"name": str
    :param binary:
        - "t2w": (B, 1, W, H, D)
        - "seg": (B, 9, W, H, D)
        - "ddf": (B, 3, W, H, D)
    :param vis_path:
    :return:
    """
    affine = np.array([[0.75, 0, 0, 0], [0, 0.75, 0, 0], [0, 0, 2.5, 0], [0, 0, 0, 1]])
    sz = img["t2w"].shape
    for i, n in enumerate(img["name"]):
        nib_img = nib.Nifti1Image(
            img["t2w"][i].reshape(*sz[-3:]).detach().cpu().numpy().astype(dtype=np.float32),
            affine=affine
        )
        nib.save(nib_img, f"{vis_path}/{n}_t2w.nii")

        nib_img = nib.Nifti1Image(
            img["seg"][i].reshape(*sz[-3:]).detach().cpu().numpy().astype(dtype=np.float32),
            affine=affine
        )
        nib.save(nib_img, f"{vis_path}/{n}_seg.nii")

        nib_img = nib.Nifti1Image(
            binary["t2w"][i].reshape(*sz[-3:]).detach().cpu().numpy().astype(dtype=np.float32),
            affine=affine
        )
        nib.save(nib_img, f"{vis_path}/{n}_registerd_t2w.nii")

        nib_img = nib.Nifti1Image(
            torch.argmax(binary["seg"][i], dim=0).reshape(*sz[-3:]).detach().cpu().numpy().astype(dtype=np.float32),
            affine=affine
        )
        nib.save(nib_img, f"{vis_path}/{n}_registerd_seg.nii")


def choose_sample():



if __name__ == '__main__':
    main()