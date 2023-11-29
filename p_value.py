import numpy as np
import torch
from scipy.stats import stats
from utils.train_eval_utils import get_parser, get_save_dir


def compute_novel_base_p_value(
        path_1="new_ckpt/img128*128*24_vit_labeledonly0.1",
        path_2="new_ckpt/img128*128*24_vit_semi0.1_1e-06_0.997_same_aug1.0_cutratio[0.1, 0.2]"
):
    d_1 = torch.load(f"{path_1}/dice_result_dict.pth", map_location=torch.device('cpu'))
    # [cls][name]["N/A"]
    list_1 = [[v["N/A"] for v in d_1[cls].values()] for cls in range(1, 9)]  # (8, num_patients)
    list_1 = np.mean(np.array(list_1), axis=0)  # (num_patients)

    d_2 = torch.load(f"{path_2}/dice_result_dict.pth", map_location=torch.device('cpu'))
    # [cls][name]["N/A"]
    list_2 = [[v["N/A"] for v in d_2[cls].values()] for cls in range(1, 9)]  # (8, num_patients)
    list_2 = np.mean(np.array(list_2), axis=0)  # (num_patients)

    p_value = stats.ttest_rel(list_1, list_2)
    print(p_value)


if __name__ == '__main__':
    print("0.1, semi")
    compute_novel_base_p_value()
    print("0.2, semi")
    compute_novel_base_p_value(
        path_1="new_ckpt/img128*128*24_vit_labeledonly0.2",
        path_2="new_ckpt/img128*128*24_vit_semi0.2_1e-06_0.997_same_aug1.0_cutratio[0.1, 0.2]"
    )
    print("0.5, semi")
    compute_novel_base_p_value(
        path_1="new_ckpt/img128*128*24_vit_semi0.5_1e-06_0.997_same_aug1.0_cutratio[0.1, 0.2]",
        path_2="new_ckpt/img128*128*24_vit_labeledonly1.0"
    )
    print("augmentation")
    compute_novel_base_p_value(
        path_1="new_ckpt/img128*128*24_vit_semi0.1_1e-06_0.997_same_aug0.0_cutratio[0.0, 0.0]",
        path_2="new_ckpt/img128*128*24_vit_semi0.1_1e-06_0.997_same_aug1.0_cutratio[0.1, 0.2]"
    )