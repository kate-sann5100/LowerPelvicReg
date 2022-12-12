import os
from collections import OrderedDict
from itertools import cycle

import torch
from monai.transforms import Spacingd
from torch.backends import cudnn
from torch.cuda import device_count, reset_peak_memory_stats, max_memory_allocated
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset import SemiDataset
from model.registration_model import Registration, ConsistencyLoss
from utils.meter import LossMeter, SemiLossMeter, DiceMeter, HausdorffMeter
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

    # initialise training dataloaders
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
    print(f"labelled dataset of size {len(l_loader)}")
    print(f"unlabelled dataset of size {len(ul_loader)}")

    # initialise validation dataloader
    val_dataset = SemiDataset(args=args, mode="val", label=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    # if over-fit, use the first validation pair for both training and validation
    if args.overfit:
        for moving, fixed in val_loader:
            l_overfit_moving, l_overfit_fixed = moving, fixed
            ul_overfit_moving, ul_overfit_fixed = moving.copy(), fixed.copy()
            del ul_overfit_moving["seg"]
            del ul_overfit_fixed["seg"]
            break
    else:
        l_overfit_moving, l_overfit_fixed, ul_overfit_moving, ul_overfit_fixed = None, None, None, None

    # initialise models
    student = torch.nn.DataParallel(
        Registration(args).cuda()
    )
    teacher = {
        teacher_id: torch.nn.DataParallel(Registration(args).cuda())
        for teacher_id in range(2)
    }

    # warm up student and teacher models
    # warm_up_save_dir = get_save_dir(args, warm_up=True)
    # if not os.path.exists(f"warm_up_save_dir/student_ckpt.pth"):
    #     warm_up(args, student, teacher, l_loader, val_loader, warm_up_save_dir)
    # student.load_state_dict(
    #     torch.load(f"{warm_up_save_dir}/student_ckpt.pth")["model"],
    #     strict=True
    # )
    # for t_id, t_model in teacher.items():
    #     t_model.load_state_dict(
    #         torch.load(f"{warm_up_save_dir}/t_{t_id}_ckpt.pth")["model"],
    #         strict=True
    #     )
    #     t_model.eval()

    # initialise student optimiser
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

        # zip labeled and unlabeled datasets
        dataloader = iter(zip(cycle(l_loader), ul_loader))
        # alternate training teacher each epoch
        curr_teacher_id = 0 if epoch % 2 != 0 else 1

        student.train()
        for step, (l, ul) in enumerate(dataloader):
            reset_peak_memory_stats()
            step_count += 1

            # load and cuda data
            l_moving, l_fixed = l
            ul_moving, ul_fixed = ul
            if args.overfit:
                l_moving, l_fixed = l_overfit_moving, l_overfit_fixed
                ul_moving, ul_fixed = ul_overfit_moving, ul_overfit_fixed
            cuda_batch(l_moving)
            cuda_batch(l_fixed)
            cuda_batch(ul_moving)
            cuda_batch(ul_fixed)

            # # backprop on labelled data
            # l_loss_dict = student(l_moving, l_fixed, semi_supervision=False)
            # print(l_loss_dict)
            # l_loss = 0
            # for k, v in l_loss_dict.items():
            #     l_loss_dict[k] = torch.mean(v)
            #     if k in ["label", "reg"]:
            #         l_loss = l_loss + torch.mean(v)
            # l_loss_meter.update(l_loss_dict)
            # optimiser.zero_grad()
            # l_loss.backward()
            # optimiser.step()

        #     # backprop on unlabelled data
        #     if args.semi_supervision:
        #         with torch.no_grad():
        #             # TODO: not support multi-gpu
        #             # TODO: divide ul and l to separate gpu
        #             ul_t_pred = [
        #                 v(ul_moving, ul_fixed, semi_supervision=True, semi_mode="train")
        #                 for _, v in teacher.items()
        #             ]
        #             ul_t_pred = torch.stack(ul_t_pred, dim=-1)
        #             ul_t_pred = torch.mean(ul_t_pred, dim=-1)
        #         ul_s_pred = student(ul_moving, ul_fixed, semi_supervision=True, semi_mode="train")
        #         ul_loss = consistency_loss(ul_t_pred, ul_s_pred, ul[1]["affine_ddf"])
        #         ul_loss_meter.update({"semi": torch.mean(ul_loss)})
        #         optimiser.zero_grad()
        #         ul_loss = ul_loss * 0.01
        #         ul_loss.backward()
        #         optimiser.step()
        #
        #     # update teacher models
        #     with torch.no_grad():
        #         update_teacher(teacher[curr_teacher_id], student, args)
        #
            writer.add_scalar(
                tag="peak_memory", scalar_value=max_memory_allocated(), global_step=step_count
            )
        #
        #     if args.overfit:
        #         break
        #
        # if args.semi_supervision:
        #     ul_loss_meter.get_average(step_count)
        # l_loss_meter.get_average(step_count)

        # # update ckpt based on validation performance
        # ckpt = {
        #     "epoch": epoch,
        #     "step_count": step_count,
        #     "student": student.state_dict(),
        #     "teacher": {k: v.state_dict() for k, v in teacher.items()},
        #     "optimiser": optimiser.state_dict(),
        # }
        # print("validating...")
        # student_dice, teacher_dice, hausdorff_result_dict = validation(
        #     args, student, teacher, val_loader,
        #     writer=writer, step=step_count, vis=None, test=False,
        #     overfit_moving=l_overfit_moving, overfit_fixed=l_overfit_fixed
        # )
        # val_metric = teacher_dice["total"][0]
        # if val_metric > best_metric:
        #     best_metric = val_metric
        #     torch.save(ckpt, f'{save_dir}/best_ckpt.pth')


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

    # initialise meters
    student_dice_meter = DiceMeter(writer, test=test, tag="student")
    teacher_dice_meter = {
        t_id: DiceMeter(writer, test=test, tag=f"t_{t_id}")
        for t_id in teacher.keys()
    }
    teacher_dice_meter["total"] = DiceMeter(writer, test=test, tag=f"t_total")
    hausdorff_meter = HausdorffMeter(writer, test=test)

    with torch.no_grad():
        for val_step, (moving, fixed) in enumerate(loader):
            # load data
            if args.overfit:
                moving, fixed = overfit_moving, overfit_fixed
            cuda_batch(moving)
            cuda_batch(fixed)

            # student prediction
            student.eval()
            student_binary = student(moving, fixed, semi_supervision=False)
            student_dice_meter.update(
                student_binary["seg"], fixed["seg"],
                name=moving["name"], fixed_ins=fixed["ins"]
            )

            # teacher prediction
            for t_id, t_model in teacher.items():
                t_model.eval()
            teacher_pred = {
                t_id: t_model(moving, fixed, semi_supervision=True, semi_mode="eval")
                for t_id, t_model in teacher.items()
            }  # (B, 1, ...), (B, 9, ...)
            teacher_pred["total"] = {
                k: torch.mean(  # "t2w", "seg"
                    torch.stack(
                        [teacher_pred[t_id][k] for t_id in teacher_pred.keys()],
                        dim=-1
                    ),  # (B, 1, ..., num_teacher), (B, 9, ..., num_teacher)
                    dim=-1
                )  # (B, 1, ...), (B, 9, ...)
                for k in ["seg", "t2w"]
            }
            for t_id, t_pred in teacher_pred.items():
                t_pred["seg"] = torch.argmax(t_pred["seg"], dim=1, keepdim=True)  # (B, 1, ...)
                teacher_dice_meter[t_id].update(
                    t_pred["seg"], fixed["seg"],
                    name=moving["name"], fixed_ins=fixed["ins"]
                )

            if test:
                # resample to resolution = (1, 1, 1)
                spacingd = Spacingd(["pred", "gt"], pixdim=[1, 1, 1], mode="nearest")
                meta_data = {"affine": moving["t2w_meta_dict"]["affine"][0]}
                resampled = spacingd(
                    {
                        "pred": t_pred["total"]["seg"][0],
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
                    pred=teacher_pred["total"],
                )

            if args.overfit:
                break

        student_dice = student_dice_meter.get_average(step)  # (dice, dict)
        teacher_dice = {
            t_id: t_meter.get_average(step)
            for t_id, t_meter in teacher_dice_meter.items()
        }  # t_id: (dice, dict)
        if test:
            hausdorff_metric, hausdorff_result_dict = hausdorff_meter.get_average(step)
        else:
            hausdorff_result_dict = None

    return student_dice, teacher_dice, hausdorff_result_dict


def warm_up_step(model, moving, fixed, optimiser, l_loss_meter):
    l_loss_dict = model(moving, fixed, semi_supervision=False)
    l_loss = 0
    for k, v in l_loss_dict.items():
        l_loss_dict[k] = torch.mean(v)
        if k in ["label", "reg"]:
            l_loss = l_loss + torch.mean(v)
    l_loss_meter.update(l_loss_dict)
    optimiser.zero_grad()
    l_loss = l_loss
    l_loss.backward()
    optimiser.step()


def warm_up(args, student, teacher, l_loader, val_loader, save_dir):
    writer = SummaryWriter(log_dir=save_dir)
    s_optimiser = Adam(student.parameters(), lr=1e-4)
    t_optimiser = {
        t_id: Adam(t_model.parameters(), lr=1e-4)
        for t_id, t_model in teacher.items()
    }

    if args.overfit:
        for moving, fixed in val_loader:
            overfit_moving, overfit_fixed = moving, fixed
            break
    else:
        overfit_moving, overfit_fixed = None, None

    step_count = 0
    s_best_metric = 0
    t_best_metric = {t_id: 0 for t_id in teacher.keys()}
    s_l_loss_meter = LossMeter(args, writer=writer, tag="student")
    t_l_loss_meter = {
        t_id: LossMeter(args, writer, tag=f"t{t_id}")
        for t_id in teacher.keys()
    }

    for epoch in range(args.warm_up_epoch):
        print(f"-----------epoch: {epoch}----------")
        student.train()
        for k in teacher.keys():
            teacher[k].train()

        for step, (fixed, moving) in enumerate(l_loader):
            reset_peak_memory_stats()
            step_count += 1
            if args.overfit:
                moving, fixed = overfit_moving, overfit_fixed
            cuda_batch(moving)
            cuda_batch(fixed)

            warm_up_step(student, moving, fixed, s_optimiser, s_l_loss_meter)
            for t_id in teacher.keys():
                warm_up_step(teacher[t_id], moving, fixed, t_optimiser[t_id], t_l_loss_meter[t_id])

            writer.add_scalar(
                tag="peak_memory", scalar_value=max_memory_allocated(), global_step=step_count
            )

            if args.overfit:
                break

        s_l_loss_meter.get_average(step_count)
        for t_id in t_l_loss_meter.keys():
            t_l_loss_meter[t_id].get_average(step_count)

        # validate current weight
        print("validating...")
        student_dice, teacher_dice, hausdorff_result_dict = validation(
            args, student, teacher, val_loader,
            writer=writer, step=step_count, vis=None, test=False,
            overfit_moving=overfit_moving, overfit_fixed=overfit_fixed
        )

        # update ckpt for each model separately based on validation performance
        if student_dice[0] > s_best_metric:
            torch.save(
                {
                    "epoch": epoch,
                    "step_count": step_count,
                    "model": student.state_dict(),
                },
                f'{save_dir}/student_ckpt.pth'
            )

        for t_id, bm in t_best_metric.items():
            if teacher_dice[t_id][0] > bm:
                torch.save(
                    {
                        "epoch": epoch,
                        "step_count": step_count,
                        "model": teacher[t_id].state_dict(),
                    },
                    f'{save_dir}/t_{t_id}_ckpt.pth'
                )


if __name__ == '__main__':
    main()