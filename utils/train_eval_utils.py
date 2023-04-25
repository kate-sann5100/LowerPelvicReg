import argparse
import os
import random
import shutil

import torch

import numpy as np

from utils import config


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('--multi_head', action='store_true')
    parser.add_argument('--reg', action='store_true')
    parser.add_argument('--supervised', action='store_true')

    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--config', type=str, default='config/basic.yaml')
    parser.add_argument('--manual_seed', default=321, dest='manual_seed')

    parser.add_argument('--label_ratio',type=float, default=0.1)
    parser.add_argument('--diff_init', action='store_true')
    args = parser.parse_args()

    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.mask:
        cfg.input = "mask"
    cfg.overfit = args.overfit
    cfg.multi_head = args.multi_head
    cfg.reg = args.reg
    cfg.semi_supervision = not args.supervised
    cfg.overwrite = args.overwrite
    cfg.test = args.test
    cfg.vis = args.vis
    cfg.manual_seed = args.manual_seed
    cfg.label_ratio = args.label_ratio
    cfg.same_init = not args.diff_init
    print(cfg)
    return cfg


def set_seed(manual_seed):
    torch.cuda.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    random.seed(manual_seed)


def get_save_dir(args, warm_up=False):
    save_dir = f"ckpt/{args.input}"
    if args.lr != 1e-4:
        save_dir += f"_lr{args.lr}"
    if len(args.organ_list) == 8:
        organ_list = "all"
    else:
        organ_list = "&".join(args.organ_list)
    save_dir += f"_{organ_list}"
    if args.multi_head:
        save_dir += "_multihead"
    if args.reg:
        save_dir += "_reg"
    if warm_up:
        save_dir += f"_warmup{args.label_ratio}"
    else:
        if args.label_ratio < 1:
            if args.semi_supervision:
                save_dir += f"_semi{args.label_ratio}_{args.semi_co}_{args.keep_rate}"
            else:
                save_dir += f"_supervised{args.label_ratio}"
    if not warm_up:
        save_dir += "_same" if args.same_init else "_diff"
        save_dir += "_noaug" if not args.strong_aug else ""
    if args.num_teacher > 1:
        save_dir += f"_{args.num_teacher}teacher"
    if args.overfit:
        save_dir += "_overfit"
    print(save_dir)
    return save_dir


def cuda_batch(batch):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.cuda()


def save_result_dicts(save_dir, dice_result_dict, hausdorff_result_dict):
    torch.save(dice_result_dict, f"{save_dir}/dice_result_dict.pth")
    torch.save(hausdorff_result_dict, f"{save_dir}/hasudorff_result_dict.pth")


def overwrite_save_dir(args, save_dir):
    if os.path.exists(f"{save_dir}/best_ckpt.pth"):
        if args.overwrite:
            shutil.rmtree(save_dir)
        else:
            raise ValueError(f"already exists: {save_dir}")