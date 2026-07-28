from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission_materials"
OUT.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(13.28, 5.31), dpi=100)
ax.set_xlim(0, 13.28)
ax.set_ylim(0, 5.31)
ax.axis("off")
boxes = [
    (0.35, 1.35, 2.35, 2.55, "3D asset +\ndeclared profile\n+ source locators", "#dbeafe"),
    (3.05, 1.35, 2.45, 2.55, "Three record states\nordinal score\nNA\nrejected profile", "#fee2e2"),
    (5.85, 1.35, 2.35, 2.55, "Software audit\nordinal mapping\nweight normalization\nmethod aggregation", "#dcfce7"),
    (8.55, 1.35, 2.05, 2.55, "Diagnostic demos\nmesh cues\nBlender / TripoSR\nQwen3-VL-8B", "#fef3c7"),
    (10.95, 1.35, 1.95, 2.55, "Next required step\nobject dossiers\nindependent\ncultural review", "#ede9fe"),
]
for x, y, width, height, label, color in boxes:
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=.08", fc=color, ec="#334155", lw=1.4))
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=12, weight="bold")
for index in range(len(boxes) - 1):
    x, y, width, height, *_ = boxes[index]
    next_x = boxes[index + 1][0]
    ax.add_patch(FancyArrowPatch((x + width + 0.06, y + height / 2), (next_x - 0.06, y + height / 2), arrowstyle="-|>", mutation_scale=16, lw=1.5, color="#475569"))
ax.text(6.64, 4.72, "Preliminary evidence-recording protocol for 3D costume constraints", ha="center", fontsize=17, weight="bold")
ax.text(6.64, 0.62, "Reproducible implementation audit — not cultural-authenticity validation", ha="center", fontsize=13, color="#334155")
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "graphical_abstract.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "graphical_abstract.pdf", bbox_inches="tight")
