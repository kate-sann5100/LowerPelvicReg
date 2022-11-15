import torch
from monai.transforms import Spacingd
from torch.backends import cudnn
from torch.cuda import device_count
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset import PatchDataset
from model.patch_registration_model import PatchRegistration
from utils.meter import LossMeter, DiceMeter, HausdorffMeter, MSEMeter
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

    train_dataset = PatchDataset(args=args, mode="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=device_count(),
        shuffle=True,
        drop_last=True,
        persistent_workers=False,
    )

    val_dataset = PatchDataset(args=args, mode="val")
    val_loader = DataLoader(val_dataset, batch_size=1)
    if args.overfit:
        for moving, fixed in val_loader:
            overfit_moving, overfit_fixed = moving, fixed
            break
    else:
        overfit_moving, overfit_fixed = None, None

    model = PatchRegistration(args)
    model = torch.nn.DataParallel(model.cuda())
    optimiser = Adam(model.parameters(), lr=1e-4)
    writer = SummaryWriter(log_dir=save_dir)

    num_epochs = 5000
    start_epoch = 0
    step_count = 0
    best_metric = 0
    loss_meter = LossMeter(args, writer=writer)

    for epoch in range(start_epoch, num_epochs):
        print(f"-----------epoch: {epoch}----------")

        model.train()
        for step, (moving, fixed) in enumerate(train_loader):
            if args.overfit:
                moving, fixed = overfit_moving, overfit_fixed

            optimiser.zero_grad()
            cuda_batch(moving)
            cuda_batch(fixed)
            loss_dict = model(moving, fixed)
            for k, v in loss_dict.items():
                loss_dict[k] = torch.mean(v)
            loss = loss_dict["total"]
            loss.backward()
            optimiser.step()
            loss_meter.update(loss_dict)
            step_count += 1

            if args.overfit:
                break

        loss_meter.get_average(step_count)
        ckpt = {
            "epoch": epoch,
            "step_count": step_count,
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
        }
        print("validating...")

        val_metric = validation(
            args, model, val_loader, writer, step_count,
            overfit_moving=overfit_moving, overfit_fixed=overfit_fixed
        )
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(ckpt, f'{save_dir}/best_ckpt.pth')


def val_worker(args):
    set_seed(args.manual_seed)

    save_dir = get_save_dir(args)

    val_dataset = PatchDataset(args=args, mode="test")
    val_loader = DataLoader(val_dataset, batch_size=1)

    model = PatchRegistration(args)
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


def validation(args, model, loader, writer=None, step=None, vis=None, test=False,
               overfit_moving=None, overfit_fixed=None):
    model.eval()
    mse_meter = MSEMeter(writer=writer)

    with torch.no_grad():
        for val_step, (moving, fixed) in enumerate(loader):
            if args.overfit:
                moving, fixed = overfit_moving, overfit_fixed
            cuda_batch(moving)
            cuda_batch(fixed)
            ddf, loss = model(moving, fixed)  # (1, 1, ...)
            mse_meter.update(loss["total"])
            if args.overfit:
                break

    mse_metric = mse_meter.get_average(step)

    return mse_metric


if __name__ == '__main__':
    main()