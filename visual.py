import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patheffects import withStroke
from matplotlib import ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ==========================================
#  DATI FEDAVG (CIFAR-10, 25 ROUND)
# ==========================================
rounds = np.arange(1, 26)

acc_a005 = np.array([
    14.45, 25.02, 33.59, 39.47, 44.15,
    47.42, 50.22, 52.78, 54.94, 56.90,
    58.34, 60.17, 61.23, 62.40, 63.65,
    64.84, 65.96, 67.05, 67.83, 68.63,
    69.63, 70.04, 70.55, 71.04, 71.86
])

acc_a01 = np.array([
    26.09, 41.49, 49.58, 54.05, 57.46,
    60.07, 62.98, 65.43, 67.70, 69.85,
    71.69, 73.54, 74.76, 76.03, 76.96,
    78.05, 78.44, 79.34, 80.14, 80.82,
    81.12, 81.80, 82.30, 82.49, 82.76
])

# ==========================================
#  STILE GLOBALE
# ==========================================
mpl.rcParams["figure.dpi"] = 220
mpl.rcParams["savefig.dpi"] = 220
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["font.size"] = 11  # base font un po' più grande

fig, ax = plt.subplots(figsize=(6.8, 3.6))
ax.set_facecolor("#F7F8FB")

# ==========================================
#  CELLE ALTERNATE VERTICALI
# ==========================================
for i in range(1, 26):
    if i % 2 == 0:
        ax.axvspan(i - 0.5, i + 0.5, color="#E9EDF7", alpha=0.40, zorder=0)

# ==========================================
#  GRIGLIA COMPLESSA
# ==========================================
ax.set_xlim(1, 25)
ax.set_ylim(10, 85)

ax.set_xticks(np.arange(1, 26, 2))
ax.set_yticks(np.arange(10, 90, 10))

ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))

ax.grid(which="major", color="#CBD2E1", lw=0.8)
ax.grid(which="minor", color="#E5E9F2", lw=0.55, ls="--", alpha=0.7)
ax.set_axisbelow(True)

# ==========================================
#  DUE LINEE (α=0.05 e α=0.1)
# ==========================================
color_005 = "#2C587A"   # blu petrolio
color_01  = "#C45A32"   # arancione “paper”

ax.plot(rounds, acc_a005, lw=1.3, color=color_005, zorder=3)
ax.scatter(
    rounds, acc_a005,
    s=45, marker="^", facecolor="white",
    edgecolor=color_005, linewidth=1.1, zorder=4,
    label=r"$\alpha{=}0.05$"
)

ax.plot(rounds, acc_a01, lw=1.3, color=color_01, zorder=3)
ax.scatter(
    rounds, acc_a01,
    s=45, marker="s", facecolor="white",
    edgecolor=color_01, linewidth=1.1, zorder=4,
    label=r"$\alpha{=}0.1$"
)

# ==========================================
#  LABEL NUMERICI SOLO OGNI 5 ROUND
# ==========================================
for x, y in zip(rounds[::5], acc_a005[::5]):
    ax.text(
        x, y + 1.6, f"{y:.1f}",
        ha="center", fontsize=9.0, color="#334",
        path_effects=[withStroke(linewidth=2, foreground="white")]
    )

for x, y in zip(rounds[::5], acc_a01[::5]):
    ax.text(
        x, y + 1.6, f"{y:.1f}",
        ha="center", fontsize=9.0, color="#533",
        path_effects=[withStroke(linewidth=2, foreground="white")]
    )

# ==========================================
#  MINI-CHART (INSET) (20–25)
# ==========================================
axins = inset_axes(
    ax,
    width="22%",
    height="38%",
    loc="lower right",
    borderpad=1.8
)
axins.set_facecolor("#F4F6FB")

axins.set_xlim(19.5, 25.5)
axins.set_ylim(65, 85)

axins.grid(which="major", color="#D9DDE8", lw=0.6)

axins.plot(rounds, acc_a005, lw=1.0, color=color_005)
axins.scatter(
    rounds, acc_a005,
    s=30, marker="^",
    facecolor="white", edgecolor=color_005, linewidth=0.9
)

axins.plot(rounds, acc_a01, lw=1.0, color=color_01)
axins.scatter(
    rounds, acc_a01,
    s=30, marker="s",
    facecolor="white", edgecolor=color_01, linewidth=0.9
)

for spine in ["top", "right"]:
    axins.spines[spine].set_visible(False)

axins.set_xticks([20, 22, 24])
axins.set_yticks([70, 75, 80])

# ⬇️ VALORI ASSI MINI-CHART PIÙ PICCOLI
axins.tick_params(axis="both", labelsize=8)

# ==========================================
#  TITOLI, ASSI (font più grandi)
# ==========================================
ax.set_title("FedAvg on CIFAR-10 with Dirichlet client splits",
             fontsize=12, weight="normal", color="#222")

ax.set_xlabel("Communication Round", fontsize=12)
ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)

# tick label un filo più grandi nel grafico grande
ax.tick_params(axis='both', labelsize=11)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# ==========================================
#  LEGENDA A SINISTRA DEL MINI-CHART
#  (dentro l'asse principale)
# ==========================================
legend = ax.legend(
    frameon=True,
    fontsize=9,
    loc="center right",
    bbox_to_anchor=(0.70, 0.30),
    ncol=1,
    borderpad=0.5,
    handlelength=2.2,
    handletextpad=0.6
)
legend.get_frame().set_facecolor("white")
legend.get_frame().set_edgecolor("#D0D5E3")

plt.tight_layout()

# Export PDF vettoriale
plt.savefig(
    "fedavg_dirichlet_legend_left_of_inset.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.show()
