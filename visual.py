import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patheffects import withStroke
from matplotlib import ticker
from sklearn.linear_model import LinearRegression
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ===========================
#  DATA
# ===========================
rounds = np.arange(1, 11)
acc = np.array([43.78, 68.25, 76.30, 81.14, 83.18,
                85.10, 86.46, 86.86, 87.46, 88.07])

# ===========================
# GLOBAL STYLE
# ===========================
mpl.rcParams["figure.dpi"] = 220
mpl.rcParams["savefig.dpi"] = 220
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["font.size"] = 10

fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.set_facecolor("#F7F8FB")

# ================================================
# 1) CELLE ALTERNATE (vertical stripes)
# ================================================
for i in range(1, 11):
    if i % 2 == 0:
        ax.axvspan(i - 0.5, i + 0.5, color="#E9EDF7", alpha=0.45, zorder=0)

# ================================================
# 2) GRIGLIA COMPLESSA (major + minor)
# ================================================
ax.set_xlim(1, 10)
ax.set_ylim(40, 90)

ax.set_xticks(rounds)
ax.set_yticks(np.arange(40, 95, 5))

ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(2.5))

ax.grid(which="major", color="#CBD2E1", lw=0.8)
ax.grid(which="minor", color="#E5E9F2", lw=0.55, ls="--", alpha=0.7)

# ================================================
# 3) CONFIDENCE RIBBON ±0.5%
# ================================================
ci = 0.5
ax.fill_between(
    rounds, acc - ci, acc + ci,
    color="#2A628F", alpha=0.06, zorder=1
)

# ================================================
# 4) LINEA + MARKER triangolo, più sottile
# ================================================
line_color = "#2C587A"
marker_edge = "#1D3C54"

ax.plot(
    rounds, acc, lw=1.3,
    color=line_color, zorder=3
)

ax.scatter(
    rounds, acc,
    s=55, marker="^",
    facecolor="white", edgecolor=marker_edge,
    linewidth=1.1, zorder=4
)

# ================================================
# 5) TREND-LINE (regressione lineare)
# ================================================
reg = LinearRegression().fit(rounds.reshape(-1, 1), acc)
trend = reg.predict(rounds.reshape(-1, 1))

ax.plot(
    rounds, trend,
    ls="--", lw=1.1,
    color="#666666", alpha=0.7,
    zorder=2
)

# ================================================
# 6) LABEL NUMERICI STACCATI
# ================================================
for x, y in zip(rounds, acc):
    ax.text(
        x, y + 1.3, f"{y:.1f}",
        ha="center", fontsize=8.2, color="#444",
        zorder=10,
        path_effects=[withStroke(linewidth=2, foreground="white")]
    )

# ================================================
# 7) FRECCE “fast gain” (prime due salite)
# ================================================
ax.annotate(
    "", xy=(2, acc[1]), xytext=(1, acc[0]),
    arrowprops=dict(
        arrowstyle="-|>", lw=1.2,
        color="#6A90AA"
    )
)
ax.text(1.55, 62, "fast\nimprovement",
        fontsize=8, color="#555",
        ha="center")

# ================================================
# 8) INSET ZOOM (round 8–10)
# ================================================
axins = inset_axes(ax, width="28%", height="45%", loc="lower right", borderpad=1.1)

axins.set_facecolor("#F4F6FB")
axins.set_xlim(7.7, 10.1)
axins.set_ylim(85.8, 88.3)

# mini grid
axins.grid(which="major", color="#D9DDE8", lw=0.6)

# plot in inset
axins.plot(rounds, acc, lw=1.2, color=line_color)
axins.scatter(rounds, acc, s=45, marker="^", facecolor="white", edgecolor=marker_edge, linewidth=1.0)

# remove clutter
for spine in ["top", "right"]:
    axins.spines[spine].set_visible(False)

axins.set_xticks([8, 9, 10])
axins.set_yticks([86, 87, 88])

# ================================================
# TITOLI E ASSI
# ================================================
ax.set_title("FedAvg convergence on CIFAR-10",
             fontsize=11, weight="normal", color="#222")

ax.set_xlabel("Communication Round")
ax.set_ylabel("Top-1 Accuracy (%)")

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.show()
