"""Render docs/img/platform_scores.png: platform score of every submission, in submission order.
Scores are the platform's own numbers (see docs/results.md)."""
from pathlib import Path

import matplotlib.pyplot as plt

SUBMISSIONS = [  # (name, score, kept in the final version)
    ("v2", 18.81, True),
    ("v4e", 23.09, True),
    ("v6_shift", 21.63, False),
    ("v5_holdout", 21.65, False),
    ("v4e_bagging", 23.89, True),
    ("v7_resid_shift", 22.16, False),
    ("v7_resid_only", 24.68, True),
    ("v9_holdout_pick", 19.37, False),
    ("v8_resid_models", 22.48, False),
]
KEPT, REJECTED = "#2a78d6", "#898781"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"

fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
names = [n for n, _, _ in SUBMISSIONS][::-1]
scores = [s for _, s, _ in SUBMISSIONS][::-1]
colors = [KEPT if k else REJECTED for _, _, k in SUBMISSIONS][::-1]
bars = ax.barh(names, scores, color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
for bar, score in zip(bars, scores):
    ax.text(bar.get_width() + 0.25, bar.get_y() + bar.get_height() / 2, f"{score:.2f}",
            va="center", ha="left", fontsize=9, color=INK)
ax.set_xlim(0, 28)
ax.xaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#c3c2b7")
ax.tick_params(colors=MUTED, labelsize=9, length=0)
ax.set_xlabel("балл платформы (0 — бейзлайн организаторов)", color=MUTED, fontsize=9)
ax.set_title("Сдачи на платформу по порядку", loc="left", color=INK, fontsize=11, pad=26)
handles = [plt.Rectangle((0, 0), 1, 1, color=KEPT), plt.Rectangle((0, 0), 1, 1, color=REJECTED)]
ax.legend(handles, ["вошло в итоговую версию", "проверено и отклонено"], loc="lower left",
          bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9, labelcolor=INK, borderaxespad=0.2)
fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "docs" / "img" / "platform_scores.png"
fig.savefig(out, facecolor=SURFACE)
print(out)
