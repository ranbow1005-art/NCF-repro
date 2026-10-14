import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# 1.原始数据
# =========================================================

factors = [8, 16, 32, 64]

data = {
    "MovieLens(Pretrain)": {
        "paper": {
            "HR": {
                "GMF":   [0.643, 0.692, 0.712, 0.705],
                "MLP":   [0.671, 0.684, 0.692, 0.702],
                "NeuMF": [0.684, 0.707, 0.726, 0.730],
            },
            "NDCG": {
                "GMF":   [0.372, 0.416, 0.435, 0.427],
                "MLP":   [0.399, 0.410, 0.425, 0.426],
                "NeuMF": [0.403, 0.426, 0.445, 0.447],
            }
        },
        "ours": {
            "HR": {
                "GMF":   [0.643, 0.691, 0.707, 0.708],
                "MLP":   [0.658, 0.673, 0.695, 0.696],
                "NeuMF": [0.689, 0.698, 0.701, 0.716],
            },
            "NDCG": {
                "GMF":   [0.368, 0.412, 0.431, 0.432],
                "MLP":   [0.380, 0.399, 0.419, 0.420],
                "NeuMF": [0.406, 0.426, 0.426, 0.442],
            }
        },
        "ylim": {
            "HR":   (0.55, 0.75),
            "NDCG": (0.30, 0.46)
        },
        "yticks": {
            "HR":   [0.55, 0.60, 0.65, 0.70, 0.75],
            "NDCG": [0.30, 0.34, 0.38, 0.42, 0.46]
        }
    },

    "Pinterest": {
        "paper": {
            "HR": {
                "GMF":   [0.866, 0.872, 0.866, 0.861],
                "MLP":   [0.859, 0.865, 0.868, 0.869],
                "NeuMF": [0.878, 0.880, 0.879, 0.877],
            },
            "NDCG": {
                "GMF":   [0.542, 0.550, 0.543, 0.541],
                "MLP":   [0.536, 0.538, 0.542, 0.545],
                "NeuMF": [0.555, 0.558, 0.555, 0.552],
            }
        },
        "ours": {
            "HR": {
                "GMF":   [0.870, 0.862, 0.857, 0.855],
                "MLP":   [0.861, 0.865, 0.864, 0.866],
                "NeuMF": [0.876, 0.872, 0.871, 0.866],
            },
            "NDCG": {
                "GMF":   [0.546, 0.536, 0.534, 0.533],
                "MLP":   [0.535, 0.541, 0.540, 0.542],
                "NeuMF": [0.554, 0.549, 0.549, 0.548],
            }
        },
        "ylim": {
            "HR":   (0.84, 0.89),
            "NDCG": (0.52, 0.56)
        },
        "yticks": {
            "HR":   [0.84, 0.85, 0.86, 0.87, 0.88, 0.89],
            "NDCG": [0.52, 0.53, 0.54, 0.55, 0.56]
        }
    }
}


# =========================================================
# 2.全局风格
# =========================================================

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["legend.fontsize"] = 10


# 模型区分：用不同marker和linestyle
model_styles_ours_only = {
    "GMF":   {"color": "#c44e52", "marker": "o", "linestyle": "--", "linewidth": 2.0, "markersize": 6, "markerfacecolor": "white"},
    "MLP":   {"color": "#dd8452", "marker": "x", "linestyle": "-",  "linewidth": 2.0, "markersize": 7},
    "NeuMF": {"color": "#d62728", "marker": "D", "linestyle": "-",  "linewidth": 2.0, "markersize": 6},
}

# 对比图：paper统一蓝色系，ours统一红色系
# 模型用marker区分，来源用颜色区分
model_markers = {
    "GMF": "o",
    "MLP": "x",
    "NeuMF": "D"
}

model_linestyles = {
    "GMF": "--",
    "MLP": "-",
    "NeuMF": "-"
}


# =========================================================
# 3.一些辅助函数
# =========================================================

def style_axis(ax, dataset_name, metric_name, y_lim, y_ticks):
    ax.set_title(dataset_name, pad=10)
    ax.set_xlabel("Factors", fontweight="bold")
    ax.set_ylabel(metric_name, fontweight="bold")
    ax.set_xticks(factors)
    ax.set_xlim(7, 65)
    ax.set_ylim(*y_lim)
    ax.set_yticks(y_ticks)

    # 只保留y方向虚线网格，接近原论文感觉
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.5)

    # 适当加粗边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    ax.tick_params(axis='both', width=1.0)


def plot_dataset_ours_only(dataset_name, save_path):
    dataset = data[dataset_name]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))

    metrics = ["HR", "NDCG"]
    ylabels = ["HR@10", "NDCG@10"]

    for ax, metric, ylabel in zip(axes, metrics, ylabels):
        for model in ["GMF", "MLP", "NeuMF"]:
            ax.plot(
                factors,
                dataset["ours"][metric][model],
                label=model,
                **model_styles_ours_only[model]
            )

        style_axis(
            ax=ax,
            dataset_name=dataset_name,
            metric_name=ylabel,
            y_lim=dataset["ylim"][metric],
            y_ticks=dataset["yticks"][metric]
        )

        ax.legend(
            loc="lower right",
            frameon=True,
            edgecolor="black",
            fancybox=False,
            ncol=1
        )

    fig.suptitle(f"{dataset_name} (Ours Only)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_dataset_paper_vs_ours(dataset_name, save_path):
    dataset = data[dataset_name]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))

    metrics = ["HR", "NDCG"]
    ylabels = ["HR@10", "NDCG@10"]

    for ax, metric, ylabel in zip(axes, metrics, ylabels):

        # 先画paper：蓝色
        for model in ["GMF", "MLP", "NeuMF"]:
            marker = model_markers[model]
            linestyle = model_linestyles[model]

            kwargs = {
                "color": "#1f77b4",   # 蓝色：paper
                "marker": marker,
                "linestyle": linestyle,
                "linewidth": 2.0,
                "markersize": 6,
                "label": f"{model}-paper"
            }

            if model == "GMF":
                kwargs["markerfacecolor"] = "white"

            ax.plot(
                factors,
                dataset["paper"][metric][model],
                **kwargs
            )

        # 再画ours：红色
        for model in ["GMF", "MLP", "NeuMF"]:
            marker = model_markers[model]
            linestyle = model_linestyles[model]

            kwargs = {
                "color": "#d62728",   # 红色：ours
                "marker": marker,
                "linestyle": linestyle,
                "linewidth": 2.0,
                "markersize": 6,
                "label": f"{model}-ours"
            }

            if model == "GMF":
                kwargs["markerfacecolor"] = "white"

            ax.plot(
                factors,
                dataset["ours"][metric][model],
                **kwargs
            )

        style_axis(
            ax=ax,
            dataset_name=dataset_name,
            metric_name=ylabel,
            y_lim=dataset["ylim"][metric],
            y_ticks=dataset["yticks"][metric]
        )

        ax.legend(
            loc="lower right",
            frameon=True,
            edgecolor="black",
            fancybox=False,
            ncol=2
        )

    fig.suptitle(f"{dataset_name} (Paper vs Ours)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# =========================================================
# 4.生成四张图
# =========================================================

plot_dataset_ours_only(
    dataset_name="MovieLens(Pretrain)",
    save_path="movielens_ours_only.png"
)

plot_dataset_paper_vs_ours(
    dataset_name="MovieLens(Pretrain)",
    save_path="movielens_paper_vs_ours.png"
)

plot_dataset_ours_only(
    dataset_name="Pinterest",
    save_path="pinterest_ours_only.png"
)

plot_dataset_paper_vs_ours(
    dataset_name="Pinterest",
    save_path="pinterest_paper_vs_ours.png"
)

# =========================================================
# 5.MovieLens 无预训练（No Pretrain）：Paper vs Ours
#    保留前面的原始数据、绘图函数以及四张已有图片
# =========================================================

# 这里创建独立的数据集配置，不覆盖上面原有的 MovieLens 数据
data["MovieLens(No Pretrain)"] = {
    "paper": {
        "HR": {
            "GMF":   [0.643, 0.692, 0.712, 0.705],
            "MLP":   [0.671, 0.684, 0.692, 0.702],
            "NeuMF": [0.688, 0.696, 0.701, 0.705],
        },
        "NDCG": {
            "GMF":   [0.372, 0.416, 0.435, 0.427],
            "MLP":   [0.399, 0.410, 0.425, 0.426],
            "NeuMF": [0.410, 0.420, 0.425, 0.426],
        }
    },
    "ours": {
        "HR": {
            "GMF":   [0.643, 0.691, 0.707, 0.708],
            "MLP":   [0.658, 0.673, 0.695, 0.696],
            "NeuMF": [0.685, 0.698, 0.705, 0.708],
        },
        "NDCG": {
            "GMF":   [0.368, 0.412, 0.431, 0.432],
            "MLP":   [0.380, 0.399, 0.419, 0.420],
            "NeuMF": [0.409, 0.421, 0.434, 0.437],
        }
    },

    # 保持与原 MovieLens 图相同的坐标范围，便于横向比较
    "ylim": {
        "HR":   (0.55, 0.75),
        "NDCG": (0.30, 0.46)
    },

    "yticks": {
        "HR":   [0.55, 0.60, 0.65, 0.70, 0.75],
        "NDCG": [0.30, 0.34, 0.38, 0.42, 0.46]
    }
}

# 复用原有函数：蓝色 Paper、红色 Ours
# 左图 HR@10，右图 NDCG@10
plot_dataset_paper_vs_ours(
    dataset_name="MovieLens(No Pretrain)",
    save_path="movielens_nopretrain_paper_vs_ours.png"
)