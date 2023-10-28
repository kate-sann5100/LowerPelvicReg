from pylatex import Tabular, MultiRow, MultiColumn, Table

from data.dataset_utils import organ_list

labelled_ratio_list = [10, 20, 50, 100]


def generate_table_by_labelled_ratio(exp_list, metric_list):
    """
    :param exp_list: list of experiments to be reported
    :param metric_list: list of metrics to be reported
    :return:
    """
    table = table_head_by_labelled_ratio()
    add_exp_by_labelled_ratio(exp_list, metric_list, table)
    doc = Table(data=table)
    doc.generate_tex(f"./table/by_labelled_ratio.tex")


def generate_table_by_class(exp_list, metric_list):
    """
    :param exp_list: list of experiments to be reported
    :param metric_list: list of metrics to be reported
    :return:
    """
    table = table_head_by_class()
    add_exp_by_class(exp_list, metric_list, table)
    doc = Table(data=table)
    doc.generate_tex(f"./table/by_class.tex")


def table_head_by_labelled_ratio():
    # metric_list = ["Dice(%)", "95%HD(mm)"]
    col_def = 'c|' + 'c' * len(labelled_ratio_list)
    row_1 = [MultiRow(2, data="method"), MultiColumn(4, data="labelled ratio(%)")]
    row_2 = ["", *labelled_ratio_list]
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


def add_exp_by_labelled_ratio(exp_list, metric_list, table):
    """

    :param labelled_ratio: int
    :param exp_list: list of str, indicating experiments to be reported
    :param metric_list: ["Dice(%)"] or ["95%HD(mm)"] or ["Dice(%)", "95%HD(mm)"]
    :return:
    """
    for i, exp in enumerate(exp_list):
        exp_result = get_result(exp)
        # exp_result: {labelled_ratio: {metric: {cls: }}}
        row = [exp]
        row += [
            "{:.2f}({:.2f})".format(
                exp_result[labelled_ratio]["Dice(%)"]["mean"], exp_result[labelled_ratio]["95%HD(mm)"]["mean"]
            ) if len(metric_list) > 1 else "{:.2f}".format(
                exp_result[labelled_ratio][metric_list[0]]["mean"]
            )
            for labelled_ratio in labelled_ratio_list
        ]
        table.add_row(row)
        if i == len(exp_list) - 1:
            table.add_hline()


def add_exp_by_class(exp_list, metric_list, table):
    """

    :param exp_list: list of str, indicating experiments to be reported
    :param metric_list: ["Dice(%)"] or ["95%HD(mm)"] or ["Dice(%)", "95%HD(mm)"]
    :return:
    """
    for labelled_ratio in labelled_ratio_list:
        if labelled_ratio == 1.0:
            exp_list = []
        elif labelled_ratio == 0.1:
            exp_list = ["sup only", "NoAug", "warp", "RegCut", "warp+RegCut"]
        else:
            exp_list = ["sup only", "warp+RegCut"]
        for i, exp in enumerate(exp_list):
            exp_result = get_result(exp)
            # exp_result: {labelled_ratio: {metric: {cls: }}}
            row = [MultiRow(len(exp_list), data=labelled_ratio)] if i == 0 else [""]
            row += [exp]
            row += [
                "{:.2f}({:.2f})".format(
                    exp_result[labelled_ratio]["Dice(%)"][cls], exp_result[labelled_ratio]["95%HD(mm)"][cls]
                ) if len(metric_list) > 1 else "{:.2f}".format(
                    exp_result[labelled_ratio][metric_list[0]][cls]
                )
                for cls in organ_list + ["mean"]
            ]
            table.add_row(row)
            if i == len(exp_list) - 1:
                table.add_hline()


def get_result(exp):
    
    return {
        labelled_ratio: {
            metric: {cls: 0 for cls in organ_list + ["mean"]}
            for metric in ["Dice(%)", "95%HD(mm)"]
        }
        for labelled_ratio in labelled_ratio_list
    }


if __name__ == '__main__':
    exp_list = ["sup only", "NoAug", "warp", "RegCut", "warp+RegCut"]
    metric_list = ["Dice(%)", "95%HD(mm)"]
    # generate_table_by_labelled_ratio(exp_list, metric_list)
    generate_table_by_class(exp_list, metric_list)