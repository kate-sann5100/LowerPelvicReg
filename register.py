import torch
from monai.transforms import Spacingd
from torch.backends import cudnn
from torch.cuda import device_count
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset import RegDataset
from model.registration_model import Registration
from utils.meter import LossMeter, DiceMeter, HausdorffMeter
from utils.train_eval_utils import get_parser, set_seed, get_save_dir, cuda_batch, save_result_dicts
from utils.visualisation import Visualisation


def main():
    args = get_parser()
    cudnn.benchmark = False
    cudnn.deterministic = True
    set_seed(args.manual_seed)

    if args.test:
        val_worker(args)
    else:
        train_worker(args)


def train_worker(args):
    set_seed(args.manual_seed)
    save_dir = get_save_dir(args)
    print(save_dir)

    args.moving_ins = args.novel_ins
    train_dataset = RegDataset(args=args, mode="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=device_count(),
        shuffle=True,
        drop_last=True,
        persistent_workers=False,
    )

    val_dataset = RegDataset(args=args, mode="val")
    val_loader = DataLoader(val_dataset, batch_size=1)

    model = Registration(args)
    model = torch.nn.DataParallel(model.cuda())
    optimiser = Adam(model.parameters(), lr=1e-4)
    writer = SummaryWriter(log_dir=save_dir)

    num_epochs = 50
    start_epoch = 0
    step_count = 0
    best_metric = 0
    loss_meter = LossMeter(writer=writer)

    for epoch in range(start_epoch, num_epochs):
        print(f"-----------epoch: {epoch}----------")

        model.train()
        for step, (moving, fixed) in enumerate(train_loader):

            optimiser.zero_grad()
            cuda_batch(moving)
            cuda_batch(fixed)
            loss_dict = model(moving, fixed)
            loss = 0
            for k, v in loss_dict.items():
                loss_dict[k] = torch.mean(v)
                if k in ["label", "reg"]:
                    loss = loss + torch.mean(v)
            loss.backward()
            optimiser.step()
            loss_dict["total"] = loss
            loss_meter.update(loss_dict)
            step_count += 1

        loss_meter.get_average(step_count)
        ckpt = {
            "epoch": epoch,
            "step_count": step_count,
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
        }
        print("validating...")

        val_metric, _, _ = validation(args, model, val_loader, writer, step_count)
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(ckpt, f'{save_dir}/best_ckpt.pth')


def val_worker(args):
    set_seed(args.manual_seed)

    save_dir = get_save_dir(args)

    args.moving_ins = args.novel_ins
    val_dataset = RegDataset(args=args, mode="test")
    val_loader = DataLoader(val_dataset, batch_size=1)

    model = Registration(args)
    print(f"model includes {sum(p.numel() for p in model.parameters())} parameters")
    model = torch.nn.DataParallel(model.cuda())
    state_dict = torch.load(f"{save_dir}/best_ckpt.pth")["model"]
    model.load_state_dict(state_dict, strict=True)

    if args.vis:
        vis = Visualisation(save_path=f"{save_dir}/vis")
    else:
        vis = None

    _, dice_result_dict, hausdorff_result_dict = validation(
        args, model, val_loader, vis=vis, test=True
    )

    if not args.vis:
        save_result_dicts(save_dir, dice_result_dict, hausdorff_result_dict)


def validation(args, model, loader, writer=None, step=None, vis=None, test=False):
    dice_meter = DiceMeter(writer, test=test)
    hausdorff_meter = HausdorffMeter(writer, test=test)
    model.eval()

    with torch.no_grad():
        for val_step, (moving, fixed) in enumerate(loader):
            cuda_batch(moving)
            cuda_batch(fixed)
            binary = model(moving, fixed)  # (1, 1, ...)
            print(moving["name"][0], fixed["ins"][0])
            dice_meter.update(
                binary["seg"], moving["seg"],
                name=moving["name"], fixed_ins=fixed["ins"]
            )
            if test:
                # resample to resolution = (1, 1, 1)
                spacingd = Spacingd(["pred", "gt"], pixdim=[1, 1, 1], mode="nearest")
                meta_data = {"affine": moving["t2w_meta_dict"]["affine"][0]}
                resampled = spacingd(
                    {
                        "pred": binary["seg"][0],
                        "gt": moving["seg"][0],
                        "pred_meta_dict": meta_data,
                        "gt_meta_dict": meta_data.copy()
                     }
                )
                hausdorff_meter.update(
                    resampled["pred"].unsqueeze(0), resampled["gt"].unsqueeze(0),
                    name=moving["name"], fixed_ins=fixed["ins"]
                )

            if vis is not None:
                vis.vis(
                    moving=moving,
                    fixed=fixed,
                    pred=binary,
                )

        dice_metric, dice_result_dict = dice_meter.get_average(step)
        if test:
            hausdorff_metric, hausdorff_result_dict = hausdorff_meter.get_average(step)
        else:
            hausdorff_result_dict = None

    return dice_metric, dice_result_dict, hausdorff_result_dict


if __name__ == '__main__':
    main()