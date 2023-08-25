import os
import time
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
from utils.train_eval_utils import cuda_batch, set_seed, get_save_dir, overwrite_save_dir, get_parser, \
    load_warm_up_ckpt, load_weight


# TODO: test augmentation
from utils.visualisation import Visualisation


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
    l_loader = DataLoader(
        l_dataset,
        batch_size=device_count(),
        shuffle=True,
        drop_last=True,
        persistent_workers=False,
    )
    print(f"{device_count()} gpus")
    print(f"labelled dataset of size {len(l_loader)}")

    ul_dataset = SemiDataset(args=args, mode="train", label=False)
    ul_loader = DataLoader(
        ul_dataset,
        batch_size=device_count(),
        shuffle=True,
        drop_last=True,
        persistent_workers=False,
    )
    print(f"unlabelled dataset of size {len(ul_loader)}")

    # initialise validation dataloader
    val_dataset = SemiDataset(args=args, mode="val", label=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    print(f"validation dataset of size {len(val_loader)}")

    debug_vis = Visualisation(f"debug_vis")

    # if over-fit, use the first training pair for both training and validation
    l_overfit_moving, l_overfit_fixed = None, None
    ul_overfit_moving, ul_overfit_fixed = None, None
    aug_overfit_moving, aug_overfit_fixed = None, None
    print("load overfit pair")
    if args.overfit:
        dataloader = iter(zip(cycle(l_loader), ul_loader))
        for l, ul in dataloader:
            l_overfit_moving, l_overfit_fixed = l
            ul_overfit_moving, ul_overfit_fixed, aug_overfit_moving, aug_overfit_fixed = ul

    # initialise models
    student = torch.nn.DataParallel(
        Registration(args).cuda()
    )
    teacher = {
        teacher_id: torch.nn.DataParallel(Registration(args).cuda())
        for teacher_id in range(args.num_teacher)
    }
    optimiser = Adam(student.parameters(), lr=args.lr)

    # load weight
    start_epoch, step_count = 0, 0
    print("load warmup weight")
    if not args.overfit:
        warm_up_save_dir = get_save_dir(args, warm_up=True)
        # if warm up weight is not available, run warm up
        warm_up_ckpt = load_warm_up_ckpt(warm_up_save_dir, args)
        if warm_up_ckpt is None or warm_up_ckpt["epoch"] < args.warm_up_epoch - 1:
            labelled_only(args, student, teacher, l_loader, val_loader, warm_up_save_dir, warm_up_ckpt, debug_vis,
                          end_epoch=args.warm_up_epoch, train_teacher=False, save_period=0)
        else:
            start_epoch, start_step = load_weight(student, teacher, warm_up_ckpt, same_init=args.same_init)
            step_count = start_step

        if args.labelled_only:
            labelled_only_save_dir = warm_up_save_dir.replace("warmup", "labeledonly")
            labelled_only(args, student, teacher, l_loader, val_loader, labelled_only_save_dir, warm_up_ckpt, debug_vis,
                          end_epoch=5000, train_teacher=False, save_period=100)

    print("weight loaded")

    if args.label_ratio == 1:
        exit()

    writer = SummaryWriter(log_dir=save_dir)

    num_epochs = 5000
    best_metric = 0
    consistency_loss = ConsistencyLoss()
    l_loss_meter = LossMeter(args, writer=writer)
    ul_loss_meter = SemiLossMeter(args, writer=writer)
    # _ = validation(
    #     args, student, teacher, val_loader,
    #     writer=writer, step=step_count, vis=None, test=False,
    #     overfit_moving=l_overfit_moving, overfit_fixed=l_overfit_fixed
    # )
    print("start training")

    for epoch in range(start_epoch, num_epochs):
        print(f"-----------epoch: {epoch}----------")

        # zip labeled and unlabeled datasets
        dataloader = iter(zip(cycle(l_loader), ul_loader))
        # alternate training teacher each epoch
        curr_teacher_id = epoch % args.num_teacher

        student.train()
        for t_id, t_model in teacher.items():
            t_model.eval()
        for step, (l, ul) in enumerate(dataloader):
            reset_peak_memory_stats()
            step_count += 1

            # load and cuda data
            l_moving, l_fixed = l
            ul_moving, ul_fixed, aug_moving, aug_fixed = ul
            print(f"step_count={step_count}, start_step={start_step}")
            if step_count == start_step:
                debug_vis.vis(l_moving, l_fixed, prefix="labelled")
                debug_vis.vis(ul_moving, ul_fixed, prefix="unlabelled")
                debug_vis.vis(aug_moving, aug_fixed, prefix="aug_unlabelled")
                exit()
            if args.overfit:
                l_moving, l_fixed = l_overfit_moving, l_overfit_fixed
                ul_moving, ul_fixed = ul_overfit_moving, ul_overfit_fixed
                ug_moving, aug_fixed = aug_overfit_moving, aug_overfit_fixed
            cuda_batch(l_moving)
            cuda_batch(l_fixed)

            # backprop on labelled data
            l_loss_dict = student(l_moving, l_fixed, semi_supervision=False)
            l_loss = 0
            for k, v in l_loss_dict.items():
                l_loss_dict[k] = torch.mean(v)
                if k in ["label", "reg"]:
                    l_loss = l_loss + torch.mean(v)
            l_loss_meter.update(l_loss_dict)
            optimiser.zero_grad()
            l_loss.backward()
            optimiser.step()

            # backprop on unlabelled data
            if args.semi_supervision:
                cuda_batch(ul_moving)
                cuda_batch(ul_fixed)
                cuda_batch(aug_fixed)
                with torch.no_grad():
                    # TODO: not support multi-gpu
                    # TODO: divide ul and l to separate gpu
                    ul_t_ddf = [
                        v(ul_moving, ul_fixed, semi_supervision=True)
                        for _, v in teacher.items()
                    ]
                    ul_t_ddf = torch.stack(ul_t_ddf, dim=-1)
                    ul_t_ddf = torch.mean(ul_t_ddf, dim=-1)
                ul_s_ddf = student(aug_moving, aug_fixed, semi_supervision=True)
                ul_loss = consistency_loss(
                    student_aug_ddf=ul_s_ddf, teacher_ddf=ul_t_ddf,
                    affine_ddf=aug_fixed["affine_ddf"], cut_mask=aug_moving["cut_mask"]
                )
                ul_loss_meter.update({"semi": torch.mean(ul_loss)})
                optimiser.zero_grad()
                ul_loss = ul_loss * args.semi_co
                ul_loss.backward()
                optimiser.step()

            # update teacher models
            with torch.no_grad():
                update_teacher(teacher[curr_teacher_id], student, args)

            writer.add_scalar(
                tag="peak_memory", scalar_value=max_memory_allocated(), global_step=step_count
            )

            if args.overfit:
                break

        l_loss_meter.get_average(step_count)
        if args.semi_supervision:
            ul_loss_meter.get_average(step_count)

        # update ckpt based on validation performance
        ckpt = {
            "epoch": epoch,
            "step_count": step_count,
            "student": student.state_dict(),
            "teacher": {k: v.state_dict() for k, v in teacher.items()},
            "optimiser": optimiser.state_dict(),
        }
        print("validating...")
        student_dice, teacher_dice, hausdorff_result_dict = validation(
            args, student, teacher, val_loader,
            writer=writer, step=step_count, vis=None, test=False,
            overfit_moving=l_overfit_moving, overfit_fixed=l_overfit_fixed
        )
        torch.save(ckpt, f'{save_dir}/last_ckpt.pth')
        val_metric = student_dice[0]
        for k, v in teacher_dice.items():
            val_metric = max(val_metric, v[0])
        val_metric = teacher_dice["total"][0]
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
               overfit_moving=None, overfit_fixed=None, labelled_only=False):

    # initialise meters
    student_dice_meter = DiceMeter(writer, test=test, tag="student")
    teacher_dice_meter = {
        t_id: DiceMeter(writer, test=test, tag=f"t_{t_id}")
        for t_id in teacher.keys()
    } if teacher is not None else None
    if teacher is not None:
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
            student_binary = student(moving_batch=moving, fixed_batch=fixed, semi_supervision=False)
            student_binary["seg"] = torch.argmax(student_binary["seg"], dim=1, keepdim=True)  # (B, 1, ...)
            student_dice_meter.update(
                student_binary["seg"], fixed["seg"],
                name=moving["name"], fixed_ins=fixed["ins"]
            )

            # teacher prediction
            if teacher is not None:
                for t_id, t_model in teacher.items():
                    t_model.eval()
                teacher_pred = {
                    t_id: t_model(moving_batch=moving, fixed_batch=fixed,semi_supervision=False)
                    for t_id, t_model in teacher.items()
                }  # (B, 1, ...), (B, 9, ...)
                if args.num_teacher > 1:
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
                else:
                    teacher_pred["total"] = teacher_pred[0]
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
        if teacher is None:
            teacher_dice = None
        else:
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


# def labelled_only(args, student, teacher, l_loader, val_loader, save_dir,
#                   start_epoch=0, step_count=0):
#     writer = SummaryWriter(log_dir=save_dir)
#     s_optimiser = Adam(student.parameters(), lr=args.lr)
#
#     overfit_moving, overfit_fixed, epoch_period = None, None, None
#     if args.overfit:
#         for moving, fixed in val_loader:
#             overfit_moving, overfit_fixed = moving, fixed
#             break
#
#     s_best_metric = None
#     s_l_loss_meter = LossMeter(args, writer=writer, tag="student")
#     print(f"warming up with dataset of size {len(l_loader)}")
#
#     end_epoch = 5000 if labelled_only else args.warm_up_epoch
#     save_epoch = 100
#     validation_step = 500
#     for epoch in range(start_epoch, end_epoch):
#         print(f"-----------epoch: {epoch}----------")
#
#         if epoch % save_epoch == 0:
#             s_best_metric = 0
#             epoch_period = (epoch // save_epoch + 1) * save_epoch
#
#         # train
#         student.train()
#         training_start = time.time()
#
#         for step, (fixed, moving) in enumerate(l_loader):
#             reset_peak_memory_stats()
#             step_count += 1
#             if args.overfit:
#                 moving, fixed = overfit_moving, overfit_fixed
#             cuda_batch(moving)
#             cuda_batch(fixed)
#             warm_up_step(student, moving, fixed, s_optimiser, s_l_loss_meter)
#             writer.add_scalar(
#                 tag="peak_memory", scalar_value=max_memory_allocated(), global_step=step_count
#             )
#
#             if step_count % validation_step == 1:
#                 # validate current weight
#                 print("validating...")
#                 validation_start = time.time()
#                 student_dice = validation(
#                     args, student, teacher, val_loader,
#                     writer=writer, step=step_count, vis=None, test=False,
#                     overfit_moving=overfit_moving, overfit_fixed=overfit_fixed, labelled_only=True
#                 )
#                 print(f"warm up validation takes {time.time() - validation_start} seconds")
#
#                 student.train()
#
#                 # update ckpt_old for each model separately based on validation performance
#                 if student_dice[0] > s_best_metric:
#                     torch.save(
#                         {
#                             "epoch": epoch,
#                             "step_count": step_count,
#                             "model": student.state_dict(),
#                         },
#                         f'{save_dir}/student_{epoch_period}_ckpt.pth'
#                     )
#
#             if args.overfit:
#                 break
#
#         # log loss
#         s_l_loss_meter.get_average(step_count)
#         if epoch == 0:
#             print(f"labelled only training epoch takes {time.time() - training_start} seconds")


def labelled_only(args, student, teacher, l_loader, val_loader, save_dir, warm_up_ckpt, debug_vis,
                  start_epoch=0, end_epoch=5000, step_count=0,
                  train_teacher=False, save_period=0):

    writer = SummaryWriter(log_dir=save_dir)
    s_optimiser = Adam(student.parameters(), lr=args.lr)
    t_optimiser = {
        t_id: Adam(t_model.parameters(), lr=args.lr)
        for t_id, t_model in teacher.items()
    } if train_teacher else None

    if warm_up_ckpt is not None:
        start_epoch, step_count = load_weight(student, teacher, warm_up_ckpt, same_init=args.same_init)
        s_optimiser.load_state_dict(warm_up_ckpt["s_optimiser"])
        if train_teacher:
            for t_id, to in t_optimiser.items():
                to.load_state_dict(warm_up_ckpt["t_optimiser"][t_id])

    overfit_moving, overfit_fixed = None, None
    if args.overfit:
        for moving, fixed in val_loader:
            overfit_moving, overfit_fixed = moving, fixed
            break

    s_best_metric = 0
    s_l_loss_meter = LossMeter(args, writer=writer, tag="student")
    t_l_loss_meter = {
        t_id: LossMeter(args, writer, tag=f"t{t_id}")
        for t_id in teacher.keys()
    } if train_teacher else None
    print(f"warming up with dataset of size {len(l_loader)}")

    validation_step = 500
    for epoch in range(start_epoch, end_epoch):
        print(f"-----------epoch: {epoch}----------")

        # train
        student.train()
        if train_teacher:
            for k in teacher.keys():
                teacher[k].train()

        training_start = time.time()

        for step, (fixed, moving) in enumerate(l_loader):
            if step == 0:
                debug_vis.vis(moving, fixed, prefix="warmup")
            reset_peak_memory_stats()
            step_count += 1
            if args.overfit:
                moving, fixed = overfit_moving, overfit_fixed
            cuda_batch(moving)
            cuda_batch(fixed)
            warm_up_step(student, moving, fixed, s_optimiser, s_l_loss_meter)
            if train_teacher:
                for t_id in teacher.keys():
                    warm_up_step(teacher[t_id], moving, fixed, t_optimiser[t_id], t_l_loss_meter[t_id])
            writer.add_scalar(
                tag="peak_memory", scalar_value=max_memory_allocated(), global_step=step_count
            )

            if step_count % validation_step == 1:
                # validate current weight
                print("validating...")
                validation_start = time.time()
                student_dice, teacher_dice, hausdorff_result_dict = validation(
                    args, student, teacher if train_teacher else None, val_loader,
                    writer=writer, step=step_count, vis=None, test=False,
                    overfit_moving=overfit_moving, overfit_fixed=overfit_fixed
                )
                print(f"labelled only validation takes {time.time() - validation_start} seconds")

                student.train()
                if train_teacher:
                    for k in teacher.keys():
                        teacher[k].train()
                ckpt = {
                    "epoch": epoch,
                    "step_count": step_count,
                    "student": student.state_dict(),
                    "s_optimiser": s_optimiser.state_dict(),
                    "teacher": {t_id: t.state_dict() for t_id, t in teacher.items()} if train_teacher else None,
                    "t_optimiser": {t_id: to.state_dict() for t_id, to in t_optimiser.items()} if train_teacher else None,
                }
                torch.save(ckpt, f'{save_dir}/last_ckpt.pth')
                # update ckpt_old for each model separately based on validation performance
                if student_dice[0] > s_best_metric:
                    torch.save(ckpt, f'{save_dir}/best_ckpt.pth')
                if save_period != 0 and step_count % save_period == 1:
                    torch.save(ckpt, f'{save_dir}/{step_count}_ckpt.pth')
            if args.overfit:
                break

        # log loss
        s_l_loss_meter.get_average(step_count)
        if train_teacher:
            for t_id in t_l_loss_meter.keys():
                t_l_loss_meter[t_id].get_average(step_count)
        if epoch == 0:
            print(f"warm up training epoch takes {time.time() - training_start} seconds")





if __name__ == '__main__':
    main()