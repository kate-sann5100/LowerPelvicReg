import numpy as np
import torch
import nibabel as nib
from monai.metrics import DiceMetric
from monai.networks import one_hot
from torch.backends import cudnn
from torch.cuda import device_count
from torch.utils.data import DataLoader

from data.dataset import AtlasDataset
from model.registration_model import Registration
from utils.train_eval_utils import cuda_batch, get_parser, set_seed


def main():
    args = get_parser()
    cudnn.benchmark = False
    cudnn.deterministic = True
    set_seed(args.manual_seed)
    save_dir = "atlas/upper_bound"
    vis_path = "atlas/upper_bound/vis"
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
    ckpt = {}
    for iter in range(10):
        atlas, avg_ddf, var_ddf = update_atlas(
            atlas, update_dataloader, model, device_count() * args.batch_size, len(dataset), args
        )
        ckpt[f"atlas_iter{iter}"] = atlas
        ckpt[f"avg_ddf_iter{iter}"] = avg_ddf
        ckpt[f"var_ddf_iter{iter}"] = var_ddf
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
        seg_sum += seg
        seg_count += 1
    seg_avg = seg_sum / seg_count  # (1, 9, W, H, D)
    atlas, best_metric = None, 0
    for img in dataloader:
        cuda_batch(img)
        one_hot_seg = one_hot(img["seg"], num_classes=9)  # (1, 9, W, H, D)
        mean_dice = DiceMetric(include_background=False, reduction="mean")(one_hot_seg, seg_avg)
        if mean_dice > best_metric:
            atlas = img.copy()
            best_metric = mean_dice
    return atlas


def update_atlas(atlas, dataloader, model, batch_size, num_samples, args):
    """
    :param atlas:
        "t2w": (B, 1, ...)
        "seg": (B, 1, ...)
    :param dataloader:
    :param model:
    :param batch_size:
    :param num_samples:
    :param args:
    :return:
    """
    ddf_size = args.size
    all_ddf = torch.zeros(num_samples, 3, *ddf_size)
    cuda_batch(atlas)
    with torch.no_grad():
        for step, img in enumerate(dataloader):
            cuda_batch(img)
            batch_atlas = {
                "t2w": atlas["t2w"].expand(len(img["t2w"]), 1, *args.size),
                "seg": atlas["seg"].expand(len(img["t2w"]), 1, *args.size),
            }
            cuda_batch(batch_atlas)
            ddf = model(moving_batch=batch_atlas, fixed_batch=img, semi_supervision=True)  # (B, 3, W, H, D)
            all_ddf[step * batch_size: step * batch_size + len(img)] = ddf
    var_ddf, avg_ddf = torch.var_mean(all_ddf, dim=0)   # (1, 3, W, H, D)
    atlas["t2w"] = model.warp(atlas["t2w"], avg_ddf, t2w=True)
    atlas["seg"] = torch.argmax(
        model.warp(atlas["seg"], ddf, t2w=False),
        dim=1, keepdim=True)
    return atlas, avg_ddf, var_ddf


def visualise_atlas(atlas, iteration, vis_path):
    affine = np.array([[0.75, 0, 0, 0], [0, 0.75, 0, 0], [0, 0, 2.5, 0], [0, 0, 0, 1]])
    sz = atlas["t2w"].shape
    for k in ["t2w", "seg"]:
        img = nib.Nifti1Image(
            atlas[k].reshape(*sz[-3:]).detach().cpu().numpy().astype(dtype=np.float32),
            affine=affine
        )
        nib.save(img, f"{vis_path}/{iteration}_{k}.nii")


if __name__ == '__main__':
    main()