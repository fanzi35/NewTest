
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


COLORS = {
    "blue": "#4C78A8",
    "orange": "#F58518",
    "green": "#54A24B",
    "red": "#E45756",
    "gray": "#A9A9A9",
    "dark": "#333333",
}


def configure_plot_style() -> None:
    """Use a restrained style and try common Chinese fonts in order."""
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.32,
            "grid.linestyle": ":",
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"未找到汇总文件：{path}")
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("q2-summary.json 顶层必须是 JSON 对象")
    history = data.get("master_history")
    if not isinstance(history, list) or not history:
        raise ValueError("汇总文件缺少非空 master_history")
    return data


def stage_value(record: dict[str, Any], stage_name: str) -> float:
    for stage in record.get("stages", []):
        if stage.get("stage") == stage_name:
            return float(stage["objective"])
    raise ValueError(
        f"epoch={record.get('epoch')} 缺少阶段 {stage_name!r} 的目标值"
    )


def stage_time(record: dict[str, Any], stage_name: str) -> float:
    for stage in record.get("stages", []):
        if stage.get("stage") == stage_name:
            return float(stage.get("wall_time_s", 0.0))
    return 0.0


def extract_history(data: dict[str, Any]) -> dict[str, np.ndarray]:
    history = sorted(data["master_history"], key=lambda item: int(item["epoch"]))
    result = {
        "epoch": np.asarray([int(item["epoch"]) for item in history]),
        "pool": np.asarray([int(item["pool_size"]) for item in history]),
        "aircraft": np.asarray(
            [stage_value(item, "total_aircraft_time") for item in history]
        ),
        "person": np.asarray(
            [stage_value(item, "total_person_intransit") for item in history]
        ),
        "fuel": np.asarray(
            [stage_value(item, "total_fuel_scaled") / 1000.0 for item in history]
        ),
    }
    result["improved"] = np.asarray(
        [bool(item.get("improved", False)) for item in history], dtype=bool
    )
    return result


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), facecolor="white")
    fig.savefig(output_base.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def plot_convergence(data: dict[str, Any], output_dir: Path) -> None:
    """Show the primary objective, route-pool growth and relative improvements."""
    values = extract_history(data)
    epoch = values["epoch"]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)
    ax_top = axes[0]
    pool_axis = ax_top.twinx()

    objective_line = ax_top.plot(
        epoch,
        values["aircraft"],
        color=COLORS["blue"],
        marker="o",
        linewidth=2.5,
        markersize=5.5,
        label="总飞机使用时间",
        zorder=3,
    )[0]
    pool_line = pool_axis.plot(
        epoch,
        values["pool"],
        color=COLORS["gray"],
        linestyle="--",
        linewidth=2.2,
        label="路线池规模",
        zorder=2,
    )[0]

    improved_epochs = epoch[values["improved"]]
    improved_values = values["aircraft"][values["improved"]]
    if improved_epochs.size:
        ax_top.scatter(
            improved_epochs,
            improved_values,
            s=90,
            marker="*",
            color=COLORS["red"],
            edgecolor="white",
            linewidth=0.8,
            label="方案改进点",
            zorder=4,
        )
        for x, y in zip(improved_epochs, improved_values):
            ax_top.annotate(
                f"{int(round(y))} min",
                (x, y),
                xytext=(8, 8),
                textcoords="offset points",
                color=COLORS["red"],
                fontsize=9,
            )

    ax_top.set_ylabel("总飞机使用时间/min")
    pool_axis.set_ylabel("路线池规模")
    ax_top.set_title("问题二路线池扩充与主目标收敛过程")
    pool_axis.grid(False)
    handles, labels = ax_top.get_legend_handles_labels()
    ax_top.legend(
        handles + [pool_line], labels + [pool_line.get_label()], loc="upper right"
    )

    ax_bottom = axes[1]
    series = [
        ("总飞机使用时间", values["aircraft"], COLORS["blue"], "o"),
        ("人员总在途时间", values["person"], COLORS["orange"], "s"),
        ("总燃油消耗", values["fuel"], COLORS["green"], "^")
    ]
    for label, raw_values, color, marker in series:
        base = float(raw_values[0])
        relative_gain = 100.0 * (base - raw_values) / base
        ax_bottom.plot(
            epoch,
            relative_gain,
            label=label,
            color=color,
            marker=marker,
            markevery=max(1, len(epoch) // 10),
            linewidth=2.0,
            markersize=4.5,
        )

    ax_bottom.axhline(0.0, color=COLORS["dark"], linewidth=0.8, alpha=0.5)
    ax_bottom.set_xlabel("迭代轮次（epoch）")
    ax_bottom.set_ylabel("相对初始方案的改进幅度/%")
    ax_bottom.set_title("字典序前三层目标的改进幅度（数值越大表示改进越多）")
    ax_bottom.legend(loc="upper left", ncol=3)
    ax_bottom.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    save_figure(fig, output_dir / "q2_search_convergence")


def plot_operator_effectiveness(data: dict[str, Any], output_dir: Path) -> None:
    """Compare how often each ALNS operator is used and what it contributes."""
    uses_raw = data.get("operator_uses", {})
    new_raw = data.get("operator_new_patterns", {})
    if not isinstance(uses_raw, dict) or not uses_raw:
        raise ValueError("汇总文件缺少 operator_uses")
    if not isinstance(new_raw, dict):
        new_raw = {}

    operators = list(uses_raw.keys())
    uses = np.asarray([float(uses_raw[name]) for name in operators])
    new_patterns = np.asarray([float(new_raw.get(name, 0)) for name in operators])
    per_use = np.divide(
        new_patterns,
        uses,
        out=np.zeros_like(new_patterns),
        where=uses > 0,
    )

    labels_cn = {
        "merge": "合并",
        "repartition2": "二路线重分配",
        "relocate": "迁移",
        "three_to_two": "三并二",
        "swap": "交换",
    }
    labels = [labels_cn.get(name, name) for name in operators]
    x = np.arange(len(operators))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars_uses = ax.bar(
        x - width / 2,
        uses,
        width,
        color=COLORS["blue"],
        label="算子调用次数",
    )
    bars_new = ax.bar(
        x + width / 2,
        new_patterns,
        width,
        color=COLORS["orange"],
        label="新增候选路线数",
    )
    ratio_axis = ax.twinx()
    ratio_line = ratio_axis.plot(
        x,
        per_use,
        color=COLORS["red"],
        marker="o",
        linewidth=2.0,
        label="单位调用新增路线数",
    )[0]

    ax.bar_label(bars_uses, fmt="%.0f", padding=3, fontsize=9)
    ax.bar_label(bars_new, fmt="%.0f", padding=3, fontsize=9)
    for xi, ratio in zip(x, per_use):
        ratio_axis.annotate(
            f"{ratio:.2f}",
            (xi, ratio),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            color=COLORS["red"],
            fontsize=9,
        )

    ax.set_xticks(x, labels)
    ax.set_ylabel("次数/条")
    ratio_axis.set_ylabel("单位调用新增路线数")
    ax.set_title("问题二ALNS邻域算子的候选路线贡献")
    ratio_axis.grid(False)
    ax.legend(
        [bars_uses, bars_new, ratio_line],
        ["算子调用次数", "新增候选路线数", "单位调用新增路线数"],
        loc="upper left",
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "q2_operator_effectiveness")


def plot_solver_time(data: dict[str, Any], output_dir: Path) -> None:
    """Optional appendix figure: wall time of each lexicographic stage."""
    history = sorted(data["master_history"], key=lambda item: int(item["epoch"]))
    epoch = np.asarray([int(item["epoch"]) for item in history])
    stage_specs = [
        ("总飞机使用时间", "total_aircraft_time", COLORS["blue"]),
        ("人员总在途时间", "total_person_intransit", COLORS["orange"]),
        ("总燃油消耗", "total_fuel_scaled", COLORS["green"]),
        ("座位利用率", "seat_utilization_1", COLORS["red"]),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bottom = np.zeros(len(history), dtype=float)
    for label, stage_name, color in stage_specs:
        times = np.asarray([stage_time(item, stage_name) for item in history])
        ax.bar(epoch, times, bottom=bottom, color=color, label=label, width=0.75)
        bottom += times

    ax.set_xlabel("迭代轮次（epoch）")
    ax.set_ylabel("单轮求解时间/s")
    ax.set_title("问题二各字典序求解阶段的计算耗时")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(ncol=2, loc="upper left")
    fig.tight_layout()
    save_figure(fig, output_dir / "q2_lexicographic_stage_time")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="根据 q2-summary.json 绘制问题二论文图"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=script_dir / "docs" / "reference_formats" / "q2-summary.json",
        help="q2-summary.json 路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "outputs" / "figures",
        help="图片输出目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_plot_style()
    data = load_summary(args.summary.resolve())
    output_dir = args.output_dir.resolve()

    plot_convergence(data, output_dir)
    plot_operator_effectiveness(data, output_dir)
    plot_solver_time(data, output_dir)

    print(f"已读取：{args.summary.resolve()}")
    print(f"图片已输出到：{output_dir}")
    for name in (
        "q2_search_convergence",
        "q2_operator_effectiveness",
        "q2_lexicographic_stage_time",
    ):
        print(f"  {name}.png / {name}.pdf")


if __name__ == "__main__":
    main()
