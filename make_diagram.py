import os

from PIL import Image
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
import numpy as np
import torch

from monai.metrics import DiceMetric
from monai.networks import one_hot
from monai.networks.blocks import Warp
from torch.backends import cudnn
from torch.cuda import device_count
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data.dataset import SemiDataset
from model.registration_model import Registration
from utils.train_eval_utils import cuda_batch, get_parser, set_seed
from utils.visualisation import Visualisation


def main():
    args = get_parser()
    print('----------------------------------')
    print(args)
    cudnn.benchmark = False
    cudnn.deterministic = True
    set_seed(args.manual_seed)
    vis = Visualisation(save_path="make_diagram")
    if os.path.exists(f"make_diagram/best_ckpt.pth"):
        ckpt = torch.load(f"make_diagram/best_ckpt.pth")
        # regcut(ckpt["moving"], ckpt["fixed"], ckpt["warped"])
        _, _, _, _, aug_fixed = get_data(args)
        warp_ddf(ckpt["moving"], ckpt["fixed"], ckpt["warped"], aug_fixed)
    else:
        l_moving, l_fixed, ul_moving, ul_fixed, aug_fixed = get_data(args)
        register(args, l_moving, l_fixed, vis)


def fake_ddf():
    x, y, z = np.meshgrid(np.arange(0, 250, 0.2),
                          np.arange(0, 250, 0.2),
                          np.arange(0, 50, 0.8))

    u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
         np.sin(np.pi * z))
    u, v, w = torch.tensor(u), torch.tensor(v), torch.tensor(w)
    ddf = torch.stack([u, v, w], dim=0).unsqueeze(0)
    # ddf = ddf[0, 0, :256, :256, :40]
    ddf = ddf[0, 0, :10, :10, :5]
    return ddf


def warp_ddf(moving, fixed, warped, aug_fixed):
    """
    :param moving:
        "t2w": (B, 1, ...)
        "seg": (B, 1, ...)
        "name": str
        "ins": int
    :param fixed:
        "t2w": (B, 1, ...)
        "seg": (B, 1, ...)
        "name": str
        "ins": int
    :param warped:
        "t2w": (B, 1, ...)
        "seg": (B, 1, ...)
        "ddf": (B, 3, ...)
    :param aug_fixed:
        "ddf": (B, 3, ...)
    :return:
    """
    # for k, v in aug_fixed.items():
    #     print(k)
    ddf = warped["ddf"]
    augmentation = aug_fixed["affine_ddf"].to(ddf)
    aug_ddf = augmentation.to(ddf) + Warp()(ddf, augmentation.to(ddf))
    # plot_ddf(ddf, "make_diagram/ddf.png")
    plot_ddf(augmentation, "make_diagram/augmentation.png")
    plot_ddf(aug_ddf, "make_diagram/warped_ddf.png")

    aug_fixed["seg"] = Warp(mode="bilinear")(fixed["seg"], augmentation),
    aug_fixed["name"] = [f"aug_{n}" for n in aug_fixed["name"]]
    aug_warp = {
        "t2w": Warp(mode="bilinear")(moving["t2w"], aug_ddf),
        "seg": Warp(mode="nearest")(moving["seg"], aug_ddf)
    }

    warp = {
        "t2w": Warp(mode="bilinear")(moving["t2w"], ddf),
        "seg": Warp(mode="nearest")(moving["seg"], ddf)
    }

    vis = Visualisation(save_path="make_diagram")
    moving_name = moving["name"]
    vis.vis(
        moving=moving,
        fixed=aug_fixed,
        pred=aug_warp,
    )
    moving["name"] = moving_name
    vis.vis(
        moving=moving,
        fixed=fixed,
        pred=warp,
    )


def regcut(moving, fixed, warped):
    """

    :param moving:
        "t2w": (B, 1, ...)
        "seg": (B, 1, ...)
        "name": str
        "ins": int
    :param fixed:
        "t2w": (B, 1, ...)
        "seg": (B, 1, ...)
        "name": str
        "ins": int
    :param warped:
        "t2w": (B, 1, ...)
        "seg": (B, 1, ...)
        "ddf": (B, 3, ...)
    :return: augmented moving, augmented_warped, ddf, augmented ddf
    """
    r_x, r_y, r_z = 100, 100, 0
    r_w, r_h, r_d = 100, 100, 20
    x, y, z = np.meshgrid(
        np.arange(0, 256, 1), np.arange(0, 256, 1), np.arange(0, 40, 1)
    )
    x[x < r_x] = 0
    x[x > r_x + r_w] = 0
    y[y < r_y] = 0
    y[y > r_y + r_h] = 0
    z[z < r_z] = 0
    z[z > r_z + r_d] = 0
    augmentation_mask = x * y * z
    augmentation_mask = torch.tensor(augmentation_mask)
    augmentation_mask[augmentation_mask > 0] = 1
    augmentation_mask = augmentation_mask[None, None, ...].to(moving["seg"])
    # augment moving
    aug_moving = {
        "t2w": moving["t2w"] * (1 - augmentation_mask) + fixed["t2w"] * augmentation_mask,
        "seg": moving["seg"] * (1 - augmentation_mask) + fixed["seg"] * augmentation_mask,
        "name": "augmove"
    }
    # augment ddf
    ddf = warped["ddf"]
    aug_ddf = ddf * (1 - augmentation_mask)
    # augment warp
    aug_warp = {
        "t2w": Warp(mode="bilinear")(aug_moving["t2w"], aug_ddf),
        "seg": Warp(mode="nearest")(aug_moving["seg"], aug_ddf)
    }
    # visualise ddf
    plot_ddf(ddf, "make_diagram/ddf.png")
    plot_ddf(aug_ddf, "make_diagram/aug_ddf.png")

    # visualise t2w and seg
    fixed_name = fixed["name"]
    vis = Visualisation(save_path="make_diagram")
    vis.vis(
        moving=aug_moving,
        fixed=fixed,
        pred=aug_warp,
    )
    warp = {
        "t2w": Warp(mode="bilinear")(moving["t2w"], ddf),
        "seg": Warp(mode="nearest")(moving["seg"], ddf)
    }
    fixed["name"] = fixed_name
    vis.vis(
        moving=moving,
        fixed=fixed,
        pred=warp,
    )


def plot_ddf(ddf, name):
    """
    :param ddf: (B, 3, H, W, D)
    :param name: str, name to save plot
    :return:
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    shape = (32, 32, 5)
    x, y, z = np.meshgrid(
        np.arange(0, shape[0], 1), np.arange(0, shape[1], 1), np.arange(0, shape[2], 1)
    )
    ddf = F.interpolate(ddf, mode="trilinear", size=shape)
    # print(torch.unique(ddf))
    ddf = np.asarray(ddf.cpu())[0]
    u, v, w = np.asarray(ddf)[0], np.asarray(ddf)[1], np.asarray(ddf)[2]
    u, v, w = u / shape[0], v / shape[1], w / shape[2]
    print(np.unique(u))
    print(np.unique(v))
    print(np.unique(w))
    ax.quiver(x, y, z, u, v, w, length=0.2, color='black')
    plt.axis("off")
    plt.savefig(name)


def get_data(args):
    l_dataset = SemiDataset(args=args, mode="train", label=True)
    l_loader = DataLoader(
        l_dataset,
        batch_size=device_count(),
        shuffle=True,
        drop_last=True,
        persistent_workers=False,
    )
    print(f"{device_count()} gpus")
    print(f"labelled dataset of size {len(l_loader)}")

    if args.label_ratio < 1:
        ul_dataset = SemiDataset(args=args, mode="train", label=False)
        ul_loader = DataLoader(
            ul_dataset,
            batch_size=device_count(),
            shuffle=True,
            drop_last=True,
            persistent_workers=False,
        )
        print(f"unlabelled dataset of size {len(ul_loader)}")

    for l_moving, l_fixed in l_loader:
        break
    for ul_moving, ul_fixed, aug_fixed in ul_loader:
        break
    cuda_batch(l_moving)
    cuda_batch(l_fixed)
    cuda_batch(ul_moving)
    cuda_batch(ul_fixed)

    return l_moving, l_fixed, ul_moving, ul_fixed, aug_fixed


def register(args, moving, fixed, vis):
    student = torch.nn.DataParallel(
        Registration(args).cuda()
    )
    optimiser = Adam(student.parameters(), lr=args.lr)
    best_dice = 0

    for step in range(10000):
        # if step > 10:
        #     exit()
        student.train()
        # backprop on labelled data
        l_loss_dict = student(moving, fixed, semi_supervision=False)
        l_loss = 0
        for k, v in l_loss_dict.items():
            print(f"{k}:{v}")
            l_loss_dict[k] = torch.mean(v)
            if k in ["label", "reg"]:
                l_loss = l_loss + torch.mean(v)
        print(l_loss)
        optimiser.zero_grad()
        l_loss.backward()
        optimiser.step()

        with torch.no_grad():
            student.eval()
            student_binary = student(moving_batch=moving, fixed_batch=fixed, semi_supervision=False)
            seg_one_hot = one_hot(student_binary["seg"], num_classes=9)  # (1, C, H, W, D)
            # seg_one_hot = one_hot(fixed["seg"], num_classes=9)  # (1, C, H, W, D)
            pred_one_hot = one_hot(fixed["seg"], num_classes=9)
            mean_dice = DiceMetric(
                include_background=False,
                reduction="sum_batch",
            )(y_pred=pred_one_hot, y=seg_one_hot).sum(dim=0)  # (C)
            nan = torch.isnan(mean_dice)
            mean_dice[nan] = 0
            if best_dice < torch.mean(mean_dice):
                print(mean_dice)
                best_dice = torch.mean(mean_dice)
                torch.save(
                    {
                        "model": student.state_dict(),
                        "moving": moving,
                        "fixed": fixed,
                        "warped": student_binary,
                    },
                    "make_diagram/best_ckpt.pth"
                )
                # vis.vis(
                #     moving=moving,
                #     fixed=fixed,
                #     pred=student_binary,
                # )


def crop_visulisation():
    for img in ["moving", "fixed", "warp", "aug_moving", "aug_warp"]:
        for v in ["volumn", "t2w", "seg"]:
            im = Image.open(f"make_diagram/{img}_{v}.png")
            width, height = im.size
            print(width, height)
            im_cropped = im.crop((200, 250, 900, 950))
            im_cropped.save(f"make_diagram/cropped/{img}_{v}_cropped.png")


def slicer():
    layoutManager = slicer.app.layoutManager()
    threeDWidget = layoutManager.threeDWidget(0)
    threeDView = threeDWidget.threeDView()
    threeDView.resetFocalPoint()

    layoutManager = slicer.app.layoutManager()
    threeDWidget = layoutManager.threeDWidget(0)
    threeDView = threeDWidget.threeDView()
    threeDView.yaw()


if __name__ == '__main__':
    # crop_visulisation()
    main()
    # ddf = fake_ddf()
    # plot_ddf(ddf, "make_diagram/fake_ddf.pdf")