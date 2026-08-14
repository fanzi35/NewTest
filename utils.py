from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> None:
    """确保目录存在。"""
    path.mkdir(parents=True, exist_ok=True)


def load_distance_matrix(path: Path) -> pd.DataFrame:
    """读取距离矩阵。"""
    df = pd.read_csv(path)
    return df.rename(columns={df.columns[0]: "from_id"})


def load_people_q1(path: Path) -> pd.DataFrame:
    """读取第一问需求。"""
    return pd.read_csv(path)


def save_dataframe_excel(df: pd.DataFrame, path: Path, sheet_name: str) -> None:
    """保存单工作表 Excel。"""
    ensure_dir(path.parent)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def save_multi_sheet_excel(sheets: dict[str, pd.DataFrame], path: Path) -> None:
    """保存多工作表 Excel。"""
    ensure_dir(path.parent)
    wb = Workbook()
    ws = wb.active
    wb.remove(ws)
    for name, df in sheets.items():
        ws = wb.create_sheet(title=name[:31])
        ws.append(list(df.columns))
        for row in df.itertuples(index=False):
            ws.append(list(row))
    wb.save(path)


def setup_matplotlib_chinese() -> None:
    """设置中文显示。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def classical_mds(distance_df: pd.DataFrame, dimensions: int = 2) -> pd.DataFrame:
    """根据距离矩阵生成二维嵌入坐标。"""
    labels = distance_df["from_id"].tolist()
    matrix = distance_df.drop(columns=["from_id"]).to_numpy(dtype=float)
    squared = matrix ** 2
    n = squared.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ squared @ centering
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    positive = np.maximum(eigvals[:dimensions], 0.0)
    coords = eigvecs[:, :dimensions] * np.sqrt(positive)
    out = pd.DataFrame(coords, columns=["x", "y"])
    out.insert(0, "node_id", labels)
    return out


def safe_gap(ub: float, lb: float) -> float:
    """计算 Gap 百分比。"""
    if ub <= 0:
        return 0.0
    return max(0.0, (ub - lb) / ub * 100.0)
