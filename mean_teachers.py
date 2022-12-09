from collections import OrderedDict
from itertools import cycle

import torch
from monai.transforms import Spacingd
from torch.backends import cudnn
from torch.cuda import device_count
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset import SemiDataset
from model.registration_model import Registration, ConsistencyLoss
from utils.meter import LossMeter, SemiLossMeter, DiceMeter, HausdorffMeter, StudentDiceMeter
from utils.train_eval_utils import cuda_batch, set_seed, get_save_dir, overwrite_save_dir, get_parser

# TODO: test augmentation


def main():
    args = get_parser()
    cudnn.benchmark = False
    cudnn.deterministic = True
    set_seed(args.manual_seed)

    if args.test:
        raise NotImplementedError
        # val_worker(args)
    else:
        train_worker(args)


def train_worker(args):
    set_seed(args.manual_seed)
    save_dir = get_save_dir(args)
    print(save_dir)
    overwrite_save_dir(args, save_dir)

    l_dataset = SemiDataset(args=args, mode="train", label=True)
    ul_dataset = SemiDataset(args=args, mode="train", label=False)
    l_loader = DataLoader(
        l_dataset,
        batch_size=device_count(),
        shuffle=True,
        drop_last=True,
        persistent_workers=False,
    )
    ul_loader = DataLoader(
        ul_dataset,
        batch_size=device_count(),
        shuffle=True,
        drop_last=True,
        persistent_workers=False,
    )

    val_dataset = SemiDataset(args=args, mode="val", label=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    if args.overfit:
        for moving, fixed in val_loader:
            l_overfit_moving, l_overfit_fixed = moving, fixed
            ul_overfit_moving, ul_overfit_fixed = moving.copy(), fixed.copy()
            del ul_overfit_moving["seg"]
            del ul_overfit_fixed["seg"]
            break
    else:
        l_overfit_moving, l_overfit_fixed, ul_overfit_moving, ul_overfit_fixed = None, None, None, None

    student = torch.nn.DataParallel(
        Registration(args).cuda()
    )
    teacher = {
        teacher_id: torch.nn.DataParallel(Registration(args).cuda())
        for teacher_id in range(2)
    }
    for k in teacher.keys():
        teacher[k].eval()
    optimiser = Adam(student.parameters(), lr=1e-4)
    writer = SummaryWriter(log_dir=save_dir)

    num_epochs = 5000
    start_epoch = 0
    step_count = 0
    best_metric = 0
    consistency_loss = ConsistencyLoss()
    l_loss_meter = LossMeter(args, writer=writer)
    ul_loss_meter = SemiLossMeter(args, writer=writer)

    for epoch in range(start_epoch, num_epochs):
        print(f"-----------epoch: {epoch}----------")

        dataloader = iter(zip(cycle(l_loader), ul_loader))
        curr_teacher_id = 0 if epoch % 2 != 0 else 1
        student.train()
        for step, (l, ul) in enumerate(dataloader):
            step_count += 1
            l_moving, l_fixed = l
            ul_moving, ul_fixed = ul
            if args.overfit:
                l_moving, l_fixed = l_overfit_moving, l_overfit_fixed
                ul_moving, ul_fixed = ul_overfit_moving, ul_overfit_fixed
            cuda_batch(l_moving)
            cuda_batch(l_fixed)
            cuda_batch(ul_moving)
            cuda_batch(ul_fixed)
            # predict unlabelled pair with both models

            l_loss_dict = student(l_moving, l_fixed, semi_supervision=False)
            l_loss = 0
            for k, v in l_loss_dict.items():
                l_loss_dict[k] = torch.mean(v)
                if k in ["label", "reg"]:
                    l_loss = l_loss + torch.mean(v)
            l_loss_meter.update(l_loss_dict)
            optimiser.zero_grad()
            l_loss = l_loss * 0.01
            l_loss.backward()
            optimiser.step()

            with torch.no_grad():
                # TODO: not support multi-gpu
                # TODO: divide ul and l to separate gpu
                ul_t_pred = [
                    v(ul_moving, ul_fixed, semi_supervision=True, semi_mode="train")
                    for _, v in teacher.items()
                ]
                ul_t_pred = torch.stack(ul_t_pred, dim=-1)
                ul_t_pred = torch.mean(ul_t_pred, dim=-1)
            ul_s_pred = student(ul_moving, ul_fixed, semi_supervision=True, semi_mode="train")
            ul_loss = consistency_loss(ul_t_pred, ul_s_pred, ul[1]["affine_ddf"])
            ul_loss_meter.update({"semi": torch.mean(ul_loss)})
            optimiser.zero_grad()
            ul_loss.backward()
            optimiser.step()

            print(ul_loss, l_loss)

            with torch.no_grad():
                update_teacher(teacher[curr_teacher_id], student, args)

            if args.overfit:
                break

        ul_loss_meter.get_average(step_count)
        l_loss_meter.get_average(step_count)
        ckpt = {
            "epoch": epoch,
            "step_count": step_count,
            "model": student.state_dict(),
            "optimiser": optimiser.state_dict(),
        }
        print("validating...")

        val_metric, _, _ = validation(
            args, student, teacher, val_loader,
            writer=writer, step=step_count, vis=None, test=False,
            overfit_moving=l_overfit_moving, overfit_fixed=l_overfit_fixed
        )
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(ckpt, f'{save_dir}/best_ckpt.pth')


def update_teacher(teacher, student, args):
    """
    update teacher model through EMA (exponential moving average)
    :param teacher: nn.Module
    :param student: nn.Module
    :param args: args
    :return:
    """
    student_state_dict = student.state_dict()
    new_teacher_state_dict = OrderedDict({
        k: student_state_dict[k] * (1 - args.keep_rate) + v * args.keep_rate
        for k, v in teacher.state_dict().items()
    })
    teacher.load_state_dict(new_teacher_state_dict)


def validation(args, student, teacher, loader,
               writer=None, step=None, vis=None, test=False,
               overfit_moving=None, overfit_fixed=None):
    student_dice_meter = StudentDiceMeter(writer, test=test)
    dice_meter = DiceMeter(writer, test=test)
    hausdorff_meter = HausdorffMeter(writer, test=test)

    with torch.no_grad():
        for val_step, (moving, fixed) in enumerate(loader):
            if args.overfit:
                moving, fixed = overfit_moving, overfit_fixed
            cuda_batch(moving)
            cuda_batch(fixed)

            student.eval()
            student_binary = student(moving, fixed, semi_supervision=False)
            student_dice_meter.update(
                student_binary["seg"], fixed["seg"],
                name=moving["name"], fixed_ins=fixed["ins"]
            )

            teacher_pred = [v(moving, fixed, semi_supervision=True, semi_mode="eval")  # (B, 9, ...)
                            for _, v in teacher.items()]
            binary = {}
            for k in ["seg", "t2w"]:
                binary[k] = torch.mean(
                    torch.stack([pred[k] for pred in teacher_pred], dim=-1),  # (B, C, ..., 9)
                    dim=-1
                )  # (B, C, ...)
            binary["seg"] = torch.argmax(binary["seg"], dim=1, keepdim=True)  # (B, 1, ...)
            dice_meter.update(
                binary["seg"], fixed["seg"],
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

            if args.overfit:
                break

        student_dice_metric, student_dice_result_dict = student_dice_meter.get_average(step)
        dice_metric, dice_result_dict = dice_meter.get_average(step)
        if test:
            hausdorff_metric, hausdorff_result_dict = hausdorff_meter.get_average(step)
        else:
            hausdorff_result_dict = None

    return dice_metric, dice_result_dict, hausdorff_result_dict


if __name__ == '__main__':
    main()