import os
import numpy as np
import torch
from pylatex import Tabular, MultiRow, MultiColumn, Table, NoEscape

from data.dataset_utils import organ_list
from utils.train_eval_utils import get_save_dir, get_parser

label_ratio_percentage_list = [10, 20, 50, 100]
label_ratio_list = [0.0, 0.1, 0.2, 0.5, 1.0]
exp_list_dict = {
    0: ["NifityReg"],
    0.1: ["sup only", "NoAug", "warp", "RegCut", "warp+RegCut"],
    0.2: ["sup only", "warp+RegCut"],
    0.5: ["sup only", "warp+RegCut"],
    1.0: ["sup only"],
}


def generate_table_by_label_ratio(exp_list, metric_list):
    """
    :param exp_list: list of experiments to be reported
    :param metric_list: list of metrics to be reported
    :return:
    """
    table = table_head_by_label_ratio()
    add_exp_by_label_ratio(args, metric_list, table)
    doc = Table(data=table)
    doc.generate_tex(f"./table/{metric_list}_by_label_ratio.tex")


def generate_table_by_population(args, structures, percentiles):
    """
    :param args:
    :param structures: ["CG", "BladderMask"]
    :param percentiles: [50, 20]
    :return:
    """
    percentile_list = []
    for p in percentiles:
        percentile_list += [f"top {p}%", f"bottom {p}%"]
    table = table_head_by_population(structures, percentile_list)
    population_list = [f"{percentile[:-1]} {structure}"
                       for structure in structures for percentile in percentile_list]
    add_exp_by_population(args, population_list, table)
    doc = Table(data=table)
    doc.generate_tex(f"./table/location_variance_by_population.tex")


def generate_table_by_class(args, metric_list):
    """
    :param exp_list: list of experiments to be reported
    :param metric_list: list of metrics to be reported
    :return:
    """
    table = table_head_by_class(metric_list)
    add_exp_by_class(args, metric_list, table)
    doc = Table(data=table)
    doc.generate_tex(f"./table/{metric_list}_by_class.tex")


def table_head_by_population(structures, percentile_list):
    col_def = 'c|c|' + 'cc|' * int(len(structures) * len(percentile_list) / 2)
    row_1 = ["labelled", "method"] + [MultiColumn(len(percentile_list), data=structure) for structure in structures]
    row_2 = ["ratio (%)", ""] + percentile_list * len(structures)
    table = Tabular(col_def)
    table.add_hline()
    table.add_row(row_1)
    table.add_row(row_2)
    table.add_hline()
    return table


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


def table_head_by_class(metric_list):
    """
    :param metric_list: ["Dice(%)"] or ["95%HD(mm)"] or ["Dice(%)", "95%HD(mm)"]
    :return:
    """
    col_def = 'c|c|' + 'c' * (len(organ_list) + 1)
    metric_list = [m.split("(")[0] for m in metric_list]
    metric_str = f"{metric_list[0]}({metric_list[1]})" if len(metric_list) > 1 else metric_list[0]
    row_1 = ["labelled", MultiRow(3, data="method"),
             MultiColumn(9, data=metric_str)]
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


def add_exp_by_label_ratio(args, metric_list, table):
    """
    :param args:
    :param metric_list: ["Dice(%)"] or ["95%HD(mm)"] or ["Dice(%)", "95%HD(mm)"]
    :param table
    :return:
    """
    for i, exp in enumerate(exp_list):
        exp_result = get_result(args, metric_list)
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


def add_exp_by_population(args, population_list, table):
    """
    :param args:
    :param population_list:
    :param table:
    :return:
    """
    for label_ratio in label_ratio_list:
        if label_ratio == 0:
            continue
        args.label_ratio = label_ratio
        exp_list = exp_list_dict[label_ratio]
        for i, exp in enumerate(exp_list):
            print(exp)
            args = update_args(args, exp)
            exp_result = get_population_variance(args, population_list, new=True)  # {p:v}
            row = [MultiRow(len(exp_list), data=label_ratio * 100)] if i == 0 else [""]
            row += [exp]
            row += [
                "{:.2f}({:.2f})".format(exp_result[p], exp_result["top "+p[7:]]/exp_result[p]) if "bottom" in p else "{:.2f}".format(exp_result[p])
                for p in population_list
            ]

            table.add_row(row)
            if i == len(exp_list) - 1:
                table.add_hline()


def add_exp_by_class(args, metric_list, table):
    """
    :param args
    :param metric_list: ["Dice(%)"] or ["95%HD(mm)"] or ["Dice(%)", "95%HD(mm)"] or ["Variance"]
    :return:
    """
    for label_ratio in label_ratio_list:
        args.label_ratio = label_ratio
        exp_list = exp_list_dict[label_ratio]
        for i, exp in enumerate(exp_list):
            if exp == "NifityReg":
                exp_result = get_result(args, metric_list, niftyreg=True)
            else:
                args = update_args(args, exp)
                exp_result = get_result(args, metric_list)
            # exp_result: {label_ratio: {metric: {cls: }}}
            row = [MultiRow(len(exp_list), data=label_ratio * 100)] if i == 0 else [""]
            row += [exp]
            # row += [
            #     "{:.2f}({:.2f})".format(
            #         exp_result["Dice(%)"][cls], exp_result["95%HD(mm)"][cls]
            #     ) if len(metric_list) > 1 else "{:.2f}".format(
            #         exp_result[metric_list[0]][cls]
            #     )
            #     for cls in organ_list + ["mean"]
            # ]
            row += [NoEscape(
                "{:.2f}$\pm${:.2f}".format(exp_result[metric_list[0]][cls], exp_result[f"{metric_list[0]}_std"][cls])
                if "mean" not in cls and "Variance" not in metric_list
                else "{:.2f}".format(exp_result[metric_list[0]][cls])
            ) for cls in organ_list + ["mean"]]
            table.add_row(row)
            if i == len(exp_list) - 1:
                table.add_hline()


def get_population_variance(args, population_list, new=False):
    """
    :param args:
    :param population_list:
    :return: {population: variance}
    """
    result_dict_path = get_save_dir(args, warm_up=args.labelled_only)
    path = f"atlas/{result_dict_path[9:]}/var_log.pth"
    if not os.path.exists(path):
        print(f"did not found {path}, skipped")
        return {p: 0 for p in population_list}
    else:
        print(f"loading result from {path}")
        d = torch.load(path, map_location=torch.device('cpu'))  # [name][term]
        out = {}  # {p:v}
        for p in population_list:
            name_list = get_population_name_list(d, p)
            if "ddf" not in d[name_list[0]].keys():
                print(f"ddf not stored in {path}, skipped")
                return {p: 0 for p in population_list}
            ddf_list = [d[n]["ddf"] for n in name_list]  # [(3, W, H, D)]
            population_variance = torch.stack(ddf_list, dim=0)  # (B, 3, W, H, D)
            if new:
                population_variance = get_variance(population_variance)  # (W, H, D)
            else:
                population_variance = torch.var(population_variance, dim=0)  # (3, W, H, D)
            population_variance = torch.mean(population_variance).numpy()
            out[p] = population_variance
        return out


def get_population_name_list(variance_dict, population):
    if population == "all":
        return list(variance_dict.keys())
    else:
        top, percentile, cls = population.split(" ")
        top = top == "top"
        percentile = int(percentile)
        name_list = list(variance_dict.keys())
        volume_list = torch.tensor([variance_dict[n][f"{cls}_volume"] for n in name_list])
        volume_list, indices = torch.sort(volume_list)
        name_list = [name_list[i] for i in indices]
        num = int(len(name_list) * percentile / 100)
        name_list = name_list[-num:] if top else name_list[:num]
        return name_list


def get_result(args, metric_list, niftyreg=False):
    if niftyreg:
        result_dict_path = "niftyreg_result"
    else:
        result_dict_path = get_save_dir(args, warm_up=args.labelled_only)
    dice_dict_path = f"{result_dict_path}/dice_result_dict.pth"
    hausdorff_dict_path = f"{result_dict_path}/hausdorff_result_dict.pth"
    variance_dict_path = f"atlas/{result_dict_path[9:]}/var_log.pth"
    population_variance_dict_path = f"atlas/{result_dict_path[9:]}/ckpt.pth"
    out = {m: {} for m in metric_list}
    for m in metric_list:
        out[f"{m}_std"] = {}
    dict_path = {
        "Dice(%)": dice_dict_path,
        "95%HD(mm)": hausdorff_dict_path,
        "Variance": variance_dict_path,
        "Population Variance": population_variance_dict_path,
    }
    for metric in metric_list:
        path = dict_path[metric]
        if os.path.exists(path):
            print(f"loading result from {path}")
            d = torch.load(path, map_location=torch.device('cpu'))
            if metric == "Population Variance":
                # [iter]["var_ddf"]: (1, 3, W, H, D)
                out[metric]["mean"] = torch.mean(d[0]["var_ddf"])
            else:
                for i, cls in enumerate(organ_list):
                    if metric == "Variance":
                        # [name][cls]
                        v = [v[f"{cls}_var"].numpy() for v in d.values()]
                    else:
                        # [cls][name]["N/A"]
                        v = [v["N/A"] for v in d[i+1].values()]
                    # if metric != "Variance":
                    #     std = np.sqrt(np.std(np.array(v)))
                    #     if "Dice" in metric:
                    #     out[f"{metric}_std"][cls] = std * 100 if metric == "Dice(%)" else std
                    v = np.array(v)
                    if niftyreg:
                        print(d)
                        print(v)
                    if metric == "Dice(%)":
                        v = v * 100
                    if metric != "Variance":
                        std = np.sqrt(np.std(np.array(v)))
                        out[f"{metric}_std"][cls] = std
                    v = np.mean(np.array(v))
                    out[metric][cls] = v
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


def update_args(args, exp):
    if exp == "sup only":
        args.labelled_only = True
    elif exp == "NoAug":
        args.labelled_only = False
        args.aug_multiplier = 0.0
        args.cut_ratio = [0.0, 0.0]
    elif exp == "warp":
        args.labelled_only = False
        args.aug_multiplier = 1.0
        args.cut_ratio = [0.0, 0.0]
    elif exp == "RegCut":
        args.labelled_only = False
        args.aug_multiplier = 0.0
        args.cut_ratio = [0.1, 0.2]
    elif exp == "warp+RegCut":
        args.labelled_only = False
        args.aug_multiplier = 1.0
        args.cut_ratio = [0.1, 0.2]
    else:
        raise ValueError(f"exp {exp} not recognised")
    return args


def get_variance(all_ddf):
    """
    :param all_ddf: (B, 3, W, H, D)
    :return:
    """
    b = all_ddf.shape[0]
    v1 = all_ddf.unsqueeze(0).repeat(b, 1, 1, 1, 1, 1)  # (B, B, 3, W, H, D)
    v2 = all_ddf.unsqueeze(1).repeat(1, b, 1, 1, 1, 1)  # (B, B, 3, W, H, D)
    diff_norm = torch.norm(v1 - v2, dim=2)  # (B, B, W, H, D)
    upper_half_mask = torch.ones(b, b)
    upper_half_mask = torch.triu(upper_half_mask, diagonal=1)  # (B, B)
    diff_norm = diff_norm * upper_half_mask[..., None, None, None].to(diff_norm)  # (B, B, W, H, D)
    sample_size = torch.sum(upper_half_mask)  # scalar
    mean = torch.sum(diff_norm, dim=(0, 1)) / sample_size  # (W, H, D)
    square_mean = torch.sum(diff_norm * diff_norm, dim=(0, 1)) / sample_size  # (W, H, D)
    variance = square_mean - mean * mean  # ( W, H, D)
    return variance


if __name__ == '__main__':
    args = get_parser()
    args.transformer = True
    exp_list = ["sup only", "NoAug", "warp", "RegCut", "warp+RegCut"]
    metric_list = ["Dice(%)", "95%HD(mm)"]
    # generate_table_by_label_ratio(exp_list, ["Population Variance"])
    # generate_table_by_population(args, ["all", "top_CG_50", "bottom_CG_50", "top_BladderMask_50", "bottom_BladderMask_50"])
    generate_table_by_population(args, ["CG", "BladderMask"], [50, 20])
    # generate_table_by_class(args, metric_list)
    # generate_table_by_class(args, metric_list[:1])
    # generate_table_by_class(args, metric_list[1:])
    # generate_table_by_class(args, ["Variance"])
