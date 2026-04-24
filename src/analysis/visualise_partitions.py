import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

def generate_partition_plot():
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    np.random.seed(42)

    num_clients = 15
    num_classes = 8
    alpha = 0.1

    iid = np.ones((num_clients, num_classes)) / num_classes
    non_iid = np.random.dirichlet([alpha] * num_classes, num_clients)

    palette = [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52",
        "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    ]

    class_labels = [f"Class {i+1}" for i in range(num_classes)]
    x = np.arange(num_clients)
    bar_w = 0.72

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"hspace": 0.3},
    )

    for ax, dist, title in zip(
        axes,
        [iid, non_iid],
        [
            r"\textbf{(a)} IID --- Uniform partitioning",
            r"\textbf{(b)} Non-IID --- Dirichlet ($\alpha=0.1$)",
        ],
    ):
        bottom = np.zeros(num_clients)
        for c in range(num_classes):
            ax.bar(
                x, dist[:, c], bottom=bottom,
                width=bar_w, color=palette[c],
                edgecolor="white", linewidth=0.4,
                label=class_labels[c] if ax is axes[0] else None,
            )
            bottom += dist[:, c]

        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"$C_{{{i+1}}}$" for i in range(num_clients)],
            fontsize=10,
        )
        ax.set_ylim(0, 1.0)
        ax.set_xlim(-0.5, num_clients - 0.5)
        ax.tick_params(axis="both", which="both", length=0)

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.yaxis.grid(True, linewidth=0.3, color="grey", alpha=0.5, linestyle="--")
        ax.set_axisbelow(True)

    axes[0].set_ylabel(r"Class proportion", fontsize=11)
    axes[1].set_ylabel(r"Class proportion", fontsize=11)

    # Shared legend along the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=num_classes,
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, -0.02),
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "partition_distribution.pdf")
    fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    generate_partition_plot()
