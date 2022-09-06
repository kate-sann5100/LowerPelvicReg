import argparse
import random
import torch

import numpy as np

from utils import config


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--fold', default=-1, dest='fold')
    parser.add_argument('--ins', default=3, dest='novel_ins')
    parser.add_argument('--shot', default=1, dest='shot')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_base_ins', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--config', type=str, default='config/fewshot.yaml')
    parser.add_argument('--manual_seed', default=321, dest='manual_seed')
    args = parser.parse_args()

    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg.fold = int(args.fold)
    cfg.novel_ins = int(args.novel_ins)
    cfg.shot = int(args.shot)
    cfg.test = args.test
    cfg.test_base_ins = args.test_base_ins
    cfg.vis = args.vis
    cfg.manual_seed = args.manual_seed
    print(cfg)
    return cfg


def set_seed(manual_seed):
    torch.cuda.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    random.seed(manual_seed)


def get_save_dir(args):
    save_dir = f"ckpt/{args.input}"
    if args.multi_head:
        save_dir += "_multihead"
    if args.reg:
        save_dir += "reg"
    return save_dir


def cuda_batch(batch):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.cuda()


def save_result_dicts(save_dir, dice_result_dict, hausdorff_result_dict):
    torch.save(dice_result_dict, f"{save_dir}/dice_result_dict.pth")
    torch.save(hausdorff_result_dict, f"{save_dir}/hasudorff_result_dict.pth")
