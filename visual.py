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

    # Style
    mpl.rcParams["figure.dpi"] = 220
    mpl.rcParams["savefig.dpi"] = 220
    mpl.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["font.size"] = 11

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.set_facecolor("#F7F8FB")

    # Alternating vertical cells
    for i in range(1, 51):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="#E9EDF7", alpha=0.35, zorder=0)

    # Grid
    ax.set_xlim(1, 50)
    ax.set_ylim(0, 100)

    ax.set_xticks(np.arange(0, 51, 5))
    ax.set_yticks(np.arange(0, 101, 10))

    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))

    ax.grid(which="major", color="#CBD2E1", lw=0.8)
    ax.grid(which="minor", color="#E5E9F2", lw=0.6, ls="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Colors
    color_05  = "#3B7F4B"   # green
    color_01  = "#C45A32"   # orange
    color_005 = "#2C587A"   # petroleum blue

    # Plot series
    def plot_series(ax, x, y, color, marker, label):
        ax.plot(x, y, lw=1.3, color=color)
        ax.scatter(x, y, s=45, marker=marker,
                   facecolor="white", edgecolor=color, linewidth=1.1,
                   label=label)

    plot_series(ax, rounds, acc_a005, color_005, "^",  r"$\alpha{=}0.50$")
    plot_series(ax, rounds, acc_a01,  color_01,  "s", r"$\alpha{=}0.10$")
    plot_series(ax, rounds, acc_a05,  color_05,  "o", r"$\alpha{=}0.05$")

    # Mini-chart
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

    # Labels and title
    ax.set_title(title, fontsize=12, color="#222")
    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Legend
    leg = ax.legend(frameon=True, fontsize=9,
                    loc="center right", bbox_to_anchor=(0.65, 0.26),
                    borderpad=0.5, handlelength=2.2, handletextpad=0.6)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("#D0D5E3")

    plt.tight_layout()
    plt.savefig(outfile, format="pdf", bbox_inches="tight")
    plt.show()


# ===========================================================
#  DATA — FEDAVG SVHN
# ===========================================================

svhn_fedavg_a005 = np.array([
    19.25,33.26,42.53,51.05,59.98,66.22,71.42,75.56,78.29,80.73,
    82.43,83.87,84.92,85.76,86.62,87.28,87.74,88.23,88.54,88.88,
    89.14,89.44,89.77,89.95,90.08,90.27,90.48,90.55,90.72,90.77,
    90.91,90.98,91.12,91.15,91.28,91.38,91.51,91.56,91.66,91.68,
    91.74,91.80,91.81,91.89,91.94,92.01,92.04,92.06,92.08,92.13
])

svhn_fedavg_a01 = np.array([
    11.94,30.65,35.39,50.91,57.45,63.21,66.76,69.88,71.92,74.16,
    75.84,77.47,78.67,79.96,80.66,81.66,82.60,83.10,83.50,84.15,
    84.56,84.72,84.87,85.28,85.49,85.93,86.08,86.25,86.17,86.62,
    86.65,86.63,86.70,86.87,86.98,87.12,87.14,87.34,87.37,87.52,
    87.64,87.48,87.56,87.45,87.73,87.72,87.72,87.78,87.72,87.71
])

svhn_fedavg_a05 = np.array([
    19.26,19.49,25.71,32.56,36.87,42.06,45.47,49.56,51.17,54.94,
    55.02,59.21,58.51,62.10,61.33,64.82,63.85,67.19,65.88,68.79,
    67.46,70.11,69.02,71.35,70.36,72.66,71.88,73.71,73.19,74.66,
    74.19,74.94,74.81,75.63,75.38,76.23,76.16,76.77,76.81,77.11,
    77.08,77.77,77.66,77.87,77.80,78.36,78.40,78.60,78.59,78.79
])

# === PLOT FedAvg SVHN ===
make_plot(
    "FedAvg on SVHN with Dirichlet client splits",
    svhn_fedavg_a05, svhn_fedavg_a01, svhn_fedavg_a005,
    "fedavg_svhn.pdf",
    inset_ylim=(70, 93)
)


# ===========================================================
#  DATA — FEDPROX SVHN
# ===========================================================

svhn_fedprox_a005 = np.array([
    19.23,33.36,42.30,51.07,60.09,66.34,71.63,75.76,78.43,81.01,
    82.39,83.80,84.90,85.68,86.64,87.28,87.88,88.22,88.60,88.96,
    89.18,89.37,89.64,89.88,90.08,90.28,90.45,90.55,90.73,90.75,
    90.91,90.98,91.15,91.26,91.39,91.53,91.54,91.59,91.67,91.73,
    91.77,91.88,91.85,91.93,91.94,92.07,92.15,92.14,92.09,92.17
])

svhn_fedprox_a01 = np.array([
    11.97,30.54,35.39,50.53,57.22,63.06,66.76,69.75,71.85,74.15,
    75.84,77.60,78.85,79.89,80.76,81.69,82.53,83.14,83.58,84.17,
    84.62,84.86,84.93,85.27,85.65,86.02,86.08,86.29,86.39,86.64,
    86.84,86.75,86.90,86.92,87.00,87.18,87.12,87.49,87.47,87.52,
    87.61,87.64,87.75,87.63,87.85,87.83,87.96,87.98,87.91,87.88
])

svhn_fedprox_a05 = np.array([
    12.99,28.80,29.56,41.04,44.40,49.46,51.93,54.14,55.96,57.84,
    59.03,60.41,61.51,62.93,63.71,64.69,65.60,66.81,67.34,68.31,
    69.13,69.48,70.30,71.05,71.39,71.95,72.02,72.86,73.10,73.46,
    73.56,73.91,74.12,74.09,74.46,74.85,74.75,74.95,75.05,75.31,
    75.34,75.57,75.56,75.73,75.98,76.07,76.01,76.33,76.10,76.30
])

# === PLOT FedProx SVHN ===
make_plot(
    "FedProx (μ=0.01) on SVHN with Dirichlet client splits",
    svhn_fedprox_a05, svhn_fedprox_a01, svhn_fedprox_a005,
    "fedprox_svhn.pdf",
    inset_ylim=(70, 93)
)
