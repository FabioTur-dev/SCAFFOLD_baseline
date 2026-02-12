import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ===========================================================
#  GENERIC PLOT FUNCTION
# ===========================================================
def make_plot(title, acc_a05, acc_a01, acc_a005, outfile, inset_ylim):

    rounds = np.arange(1, 51)

    mpl.rcParams["figure.dpi"] = 220
    mpl.rcParams["savefig.dpi"] = 220
    mpl.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["font.size"] = 11

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.set_facecolor("#F7F8FB")

    for i in range(1, 51):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="#E9EDF7", alpha=0.35, zorder=0)

    ax.set_xlim(1, 50)
    ax.set_ylim(0, 100)

    ax.set_xticks(np.arange(0, 51, 5))
    ax.set_yticks(np.arange(0, 101, 10))

    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))

    ax.grid(which="major", color="#CBD2E1", lw=0.8)
    ax.grid(which="minor", color="#E5E9F2", lw=0.6, ls="--", alpha=0.7)
    ax.set_axisbelow(True)

    color_05  = "#3B7F4B"
    color_01  = "#C45A32"
    color_005 = "#2C587A"

    def plot_series(ax, x, y, color, marker, label):
        ax.plot(x, y, lw=1.3, color=color)
        ax.scatter(x, y, s=45, marker=marker,
                   facecolor="white", edgecolor=color, linewidth=1.1,
                   label=label)

    plot_series(ax, rounds, acc_a005, color_005, "^",  r"$\alpha{=}0.50$")
    plot_series(ax, rounds, acc_a01,  color_01,  "s", r"$\alpha{=}0.10$")
    plot_series(ax, rounds, acc_a05,  color_05,  "o", r"$\alpha{=}0.05$")

    axins = inset_axes(ax, width="22%", height="38%",
                       loc="lower right", borderpad=1.8)
    axins.set_facecolor("#F4F6FB")
    axins.set_xlim(38, 50.5)
    axins.set_ylim(*inset_ylim)
    axins.grid(which="major", color="#D9DDE8", lw=0.6)

    for y, c, m in [(acc_a005, color_005, "^"),
                    (acc_a01,  color_01,  "s"),
                    (acc_a05,  color_05,  "o")]:
        axins.plot(rounds, y, lw=1.0, color=c)
        axins.scatter(rounds, y, s=28, marker=m,
                      facecolor="white", edgecolor=c, linewidth=0.9)

    for spine in ["top", "right"]:
        axins.spines[spine].set_visible(False)

    axins.set_xticks([40, 45, 50])
    axins.tick_params(axis="both", labelsize=8)

    ax.set_title(title, fontsize=12, color="#222")
    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    leg = ax.legend(frameon=True, fontsize=9,
                    loc="center right", bbox_to_anchor=(0.65, 0.26))
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("#D0D5E3")

    plt.tight_layout()
    plt.savefig(outfile, format="pdf", bbox_inches="tight")
    plt.show()


# ===========================================================
#  SCAFFOLD — SVHN
# ===========================================================

svhn_scaffold_a005 = np.array([
18.79,33.19,42.46,51.46,59.55,66.67,71.43,75.51,78.19,80.80,
82.68,84.07,85.23,85.98,86.70,87.31,87.76,88.23,88.63,89.03,
89.31,89.57,89.83,90.05,90.24,90.41,90.47,90.71,90.79,90.92,
91.03,91.15,91.21,91.31,91.28,91.42,91.54,91.52,91.57,91.65,
91.72,91.74,91.79,91.88,91.87,91.96,92.01,92.00,92.00,92.00
])

svhn_scaffold_a01 = np.array([
12.05,27.09,35.05,48.87,54.69,60.43,63.74,66.99,69.39,71.45,
73.79,75.48,77.13,78.41,79.65,80.43,81.43,82.09,82.82,83.02,
83.80,84.12,84.49,84.65,85.08,85.46,85.55,85.66,85.96,85.68,
85.99,86.35,86.68,86.58,86.67,86.76,86.76,86.97,87.02,87.14,
87.18,87.10,87.36,87.24,87.34,87.45,87.28,87.58,87.46,87.45
])

svhn_scaffold_a05 = np.array([
12.87,28.32,29.70,40.65,44.68,49.37,52.17,53.98,56.20,57.45,
58.73,60.01,61.33,62.27,63.30,64.42,65.53,66.24,67.12,68.48,
68.97,69.32,70.42,70.73,71.37,71.78,72.36,72.63,72.94,73.15,
73.66,73.69,73.77,74.42,74.39,74.51,74.69,74.91,74.96,75.34,
75.25,75.27,75.21,75.50,75.69,75.86,75.86,75.83,76.08,75.99
])

make_plot(
    "SCAFFOLD on SVHN with Dirichlet client splits",
    svhn_scaffold_a05,
    svhn_scaffold_a01,
    svhn_scaffold_a005,
    "scaffold_svhn.pdf",
    inset_ylim=(70, 93)
)


# ===========================================================
#  SCAFFOLD — CIFAR-10
# ===========================================================

cifar10_scaffold_a005 = np.array([
31.81,51.91,59.42,66.42,71.22,75.26,78.14,80.09,81.32,82.69,
84.02,84.71,85.43,86.03,86.53,86.84,87.60,87.76,88.08,88.29,
88.60,88.85,89.20,89.41,89.61,89.67,89.79,90.02,90.18,90.28,
90.20,90.46,90.49,90.60,90.77,90.81,90.80,90.79,91.00,91.15,
91.20,91.21,91.29,91.40,91.38,91.39,91.46,91.48,91.58,91.66
])

cifar10_scaffold_a01 = np.array([
27.71,44.79,55.13,61.50,66.52,70.03,72.57,75.92,77.09,79.04,
80.00,81.09,82.19,82.65,83.29,83.65,84.12,84.59,84.84,85.34,
85.53,85.65,85.76,86.38,86.32,86.56,86.71,86.89,87.06,87.14,
87.38,87.37,87.49,87.41,87.64,87.65,87.90,87.71,87.93,87.96,
87.95,88.04,88.12,87.95,88.20,88.26,88.08,88.30,88.34,88.39
])

cifar10_scaffold_a05 = np.array([
20.36,39.34,47.35,54.34,56.76,59.44,61.16,62.35,63.60,64.58,
65.48,66.65,67.19,67.90,68.46,68.96,68.99,69.60,69.92,70.51,
70.69,70.88,71.02,71.27,71.51,71.50,72.04,72.17,72.14,72.46,
72.36,72.69,72.73,72.79,72.89,72.97,72.85,73.00,73.16,73.33,
73.30,73.27,73.50,73.51,73.47,73.52,73.71,73.62,73.81,73.78
])

make_plot(
    "SCAFFOLD on CIFAR-10 with Dirichlet client splits",
    cifar10_scaffold_a05,
    cifar10_scaffold_a01,
    cifar10_scaffold_a005,
    "scaffold_cifar10.pdf",
    inset_ylim=(70, 92)
)


# ===========================================================
#  SCAFFOLD — CIFAR-100
# ===========================================================

cifar100_scaffold_a005 = np.array([
1.50,2.94,5.21,8.79,12.84,17.57,22.21,26.35,30.21,33.28,
36.76,39.59,42.17,44.68,46.95,48.66,50.16,51.65,53.13,54.26,
55.59,56.58,57.58,58.46,59.11,59.87,60.62,61.04,61.68,62.27,
62.76,63.33,63.66,63.97,64.32,64.82,65.05,65.30,65.62,66.14,
66.61,66.63,66.84,67.24,67.38,67.80,68.10,68.21,68.32,68.60
])

cifar100_scaffold_a01 = np.array([
1.63,3.12,5.98,9.98,13.68,17.65,21.59,25.40,28.12,30.96,
33.33,35.32,37.74,39.67,41.38,43.14,44.55,45.86,47.36,48.66,
49.50,50.68,51.38,52.35,53.04,54.03,54.73,55.61,56.00,56.80,
57.32,57.78,58.32,58.58,59.19,59.45,59.98,60.23,60.75,61.13,
61.53,61.81,62.20,62.35,62.79,63.18,63.40,63.63,63.95,64.09
])

cifar100_scaffold_a05 = np.array([
1.93,4.21,7.64,10.87,14.60,17.62,20.78,23.67,26.18,28.87,
30.77,32.69,35.09,36.73,38.28,40.05,41.32,42.27,43.58,44.84,
45.78,46.82,47.98,48.89,49.65,50.40,51.08,52.14,52.83,53.30,
54.05,54.78,55.22,55.87,56.26,56.70,57.09,57.56,57.92,58.45,
58.65,59.08,59.33,59.72,59.98,60.28,60.45,61.09,61.30,61.63
])

make_plot(
    "SCAFFOLD on CIFAR-100 with Dirichlet client splits",
    cifar100_scaffold_a05,
    cifar100_scaffold_a01,
    cifar100_scaffold_a005,
    "scaffold_cifar100.pdf",
    inset_ylim=(45, 70)
)
