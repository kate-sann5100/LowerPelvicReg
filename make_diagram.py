import torch
from monai.metrics import DiceMetric
from monai.networks import one_hot
from torch.backends import cudnn
from torch.cuda import device_count
from torch.optim import Adam
from torch.utils.data import DataLoader

from data.dataset import SemiDataset
from model.registration_model import Registration
from utils.train_eval_utils import cuda_batch, get_parser, set_seed


def main():
    args = get_parser()
    cudnn.benchmark = False
    cudnn.deterministic = True
    set_seed(args.manual_seed)
    l_moving, l_fixed, ul_moving, ul_fixed = get_data(args)
    register(args, l_moving, l_fixed)


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
    for ul_moving, ul_fixed in ul_loader:
        break
    cuda_batch(l_moving)
    cuda_batch(l_fixed)
    cuda_batch(ul_moving)
    cuda_batch(ul_fixed)

    return l_moving, l_fixed, ul_moving, ul_fixed


def register(args, moving, fixed):
    student = torch.nn.DataParallel(
        Registration(args).cuda()
    )
    optimiser = Adam(student.parameters(), lr=args.lr)
    best_dice = 0

    for step in enumerate(10000):
        if step > 10:
            exit()
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

        student.eval()
        student_binary = student(moving_batch=moving, fixed_batch=fixed, semi_supervision=False)
        seg_one_hot = one_hot(student_binary["seg"], num_classes=9)  # (1, C, H, W, D)
        pred_one_hot = one_hot(fixed["seg"], num_classes=9)
        mean_dice = DiceMetric(
            include_background=False,
            reduction="sum_batch",
        )(y_pred=pred_one_hot, y=seg_one_hot).sum(dim=0)  # (C)
        nan = torch.isnan(mean_dice)
        mean_dice[nan] = 0
        print(mean_dice)


if __name__ == '__main__':
    main()