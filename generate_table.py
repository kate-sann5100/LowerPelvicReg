import os
import numpy as np
import torch
from pylatex import Tabular, MultiRow, MultiColumn, Table

from data.dataset_utils import organ_list
from utils.train_eval_utils import get_save_dir, get_parser

label_ratio_percentage_list = [10, 20, 50, 100]
label_ratio_list = [0.1, 0.2, 0.5, 1.0]


def generate_table_by_label_ratio(exp_list, metric_list):
    """
    :param exp_list: list of experiments to be reported
    :param metric_list: list of metrics to be reported
    :return:
    """
    table = table_head_by_label_ratio()
    add_exp_by_label_ratio(exp_list, metric_list, table)
    doc = Table(data=table)
    doc.generate_tex(f"./table/by_label_ratio.tex")


def generate_table_by_class(args, metric_list):
    """
    :param exp_list: list of experiments to be reported
    :param metric_list: list of metrics to be reported
    :return:
    """
    table = table_head_by_class()
    add_exp_by_class(args, metric_list, table)
    doc = Table(data=table)
    doc.generate_tex(f"./table/by_class.tex")


def table_head_by_label_ratio():
    # metric_list = ["Dice(%)", "95%HD(mm)"]
    col_def = 'c|' + 'c' * len(label_ratio_percentage_list)
    row_1 = [MultiRow(2, data="method"), MultiColumn(4, data="labelled ratio(%)")]
    row_2 = ["", *label_ratio_percentage_list]
    table = Tabular(col_def)
    table.add_hline()
    table.add_row(row_1)
    table.add_row(row_2)
    table.add_hline()
    return table


def table_head_by_class():
    col_def = 'c|c|' + 'c' * (len(organ_list) + 1)
    row_1 = ["labelled", MultiRow(3, data="method"), MultiColumn(9, data="Dice(95% Hausdorff Distance)")]
    row_2 = ["ratio", "",
             MultiRow(2, data="Bladder"), MultiRow(2, data="Bone"), "Obturator", "Transition",
              "Central", MultiRow(2, data="Rectum"), "seminal", "neurovascular", MultiRow(2, data="mean")]
    row_3 = ["(%)", "",
             "", "", "internus", "zone",
              "gland", "", "vesicle", "bundle", ""]
    table = Tabular(col_def)
    table.add_hline()
    table.add_row(row_1)
    table.add_row(row_2)
    table.add_row(row_3)
    table.add_hline()
    return table


def add_exp_by_label_ratio(exp_list, metric_list, table):
    """

    :param label_ratio: int
    :param exp_list: list of str, indicating experiments to be reported
    :param metric_list: ["Dice(%)"] or ["95%HD(mm)"] or ["Dice(%)", "95%HD(mm)"]
    :return:
    """
    for i, exp in enumerate(exp_list):
        exp_result = get_result(exp)
        # exp_result: {label_ratio: {metric: {cls: }}}
        row = [exp]
        row += [
            "{:.2f}({:.2f})".format(
                exp_result[label_ratio]["Dice(%)"]["mean"], exp_result[label_ratio]["95%HD(mm)"]["mean"]
            ) if len(metric_list) > 1 else "{:.2f}".format(
                exp_result[label_ratio][metric_list[0]]["mean"]
            )
            for label_ratio in label_ratio_percentage_list
        ]
        table.add_row(row)
        if i == len(exp_list) - 1:
            table.add_hline()


def add_exp_by_class(args, metric_list, table):
    """
    :param args
    :param metric_list: ["Dice(%)"] or ["95%HD(mm)"] or ["Dice(%)", "95%HD(mm)"]
    :return:
    """
    for label_ratio in label_ratio_list:
        args.label_ratio = label_ratio
        if label_ratio == 1.0:
            exp_list = []
        elif label_ratio == 0.1:
            exp_list = ["sup only", "NoAug", "warp", "RegCut", "warp+RegCut"]
        else:
            exp_list = ["sup only", "warp+RegCut"]
        for i, exp in enumerate(exp_list):
            if exp == "sup only":
                args.labelled_only = True
            elif exp == "NoAug":
                args.labelled_only = False
                args.aug_multiplier = 0.0
                cut_ratio: [0.0, 0.0]
            elif exp == "warp":
                args.labelled_only = False
                args.aug_multiplier = 1.0
                cut_ratio: [0.0, 0.0]
            elif exp == "RegCut":
                args.labelled_only = False
                args.aug_multiplier = 0.0
                cut_ratio: [0.1, 0.2]
            elif exp == "warp+RegCut":
                args.labelled_only = False
                args.aug_multiplier = 1.0
                cut_ratio: [0.1, 0.2]
            else:
                raise ValueError(f"exp {exp} not recognised")
            exp_result = get_result(args, metric_list)
            # exp_result: {label_ratio: {metric: {cls: }}}
            row = [MultiRow(len(exp_list), data=label_ratio * 100)] if i == 0 else [""]
            row += [exp]
            row += [
                "{:.2f}({:.2f})".format(
                    exp_result["Dice(%)"][cls], exp_result["95%HD(mm)"][cls]
                ) if len(metric_list) > 1 else "{:.2f}".format(
                    exp_result[metric_list[0]][cls]
                )
                for cls in organ_list + ["mean"]
            ]
            table.add_row(row)
            if i == len(exp_list) - 1:
                table.add_hline()


def get_result(args, metric_list):
    result_dict_path = get_save_dir(args, warm_up=args.labelled_only)
    dice_dict_path = f"{result_dict_path}/dice_result_dict.pth"
    hausdorff_dict_path = f"{result_dict_path}/hausdorff_result_dict.pth"
    out = {m: {} for m in metric_list}
    dict_path = {"Dice(%)": dice_dict_path, "95%HD(mm)": hausdorff_dict_path}
    for metric in metric_list:
        path = dict_path[metric]
        if os.path.exists(path):
            print(f"loading result from {path}")
            d = torch.load(path)  #[cls][name]["N/A"]
            for i, cls in enumerate(organ_list):
                v = [v["N/A"] for v in d[i+1].values()]
                v = np.mean(np.array(v))
                out[metric][cls] = v * 100 if metric == "Dice(%)" else v
            out[metric]["mean"] = np.mean(np.array(list(out[metric].values())))
        else:
            print(f"did not found {path}, skipped")
            out[metric] = {cls: 0 for cls in organ_list + ["mean"]}
    return out
    # return {
    #     label_ratio: {
    #         metric: {cls: 0 for cls in organ_list + ["mean"]}
    #         for metric in ["Dice(%)", "95%HD(mm)"]
    #     }
    #     for label_ratio in label_ratio_percentage_list
    # }


if __name__ == '__main__':
    args = get_parser()
    args.transformer = True
    exp_list = ["sup only", "NoAug", "warp", "RegCut", "warp+RegCut"]
    metric_list = ["Dice(%)", "95%HD(mm)"]
    # generate_table_by_label_ratio(exp_list, metric_list)
    generate_table_by_class(args, metric_list)