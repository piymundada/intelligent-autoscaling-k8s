"""
Thesis figure: LSTM training and validation loss curves (Chapter 4, section 4.5.4).

Drawn from the per-epoch history saved by ml/train.py into the two model result
JSONs, so the figure always matches the deployed models.

  left  panel: Minikube forecaster  (16 hidden units, 1 layer)
  right panel: Amazon EKS forecaster (64 hidden units, 2 layers)

Usage:  /usr/bin/python3 analysis/figure_training_curves.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_PNG = "docs/figure_4_1_training_curves.png"
OUT_SVG = "docs/figure_4_1_training_curves.svg"
TRAIN_C, VAL_C, MARK_C = "#3498db", "#e74c3c", "#7f8c8d"
DPI = 150
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
# Keep SVG text as real <text> elements so every label stays editable in
# Illustrator / Inkscape / Visio rather than being converted to outlines.
plt.rcParams["svg.fonttype"] = "none"

PANELS = [
    ("ml/models/lstm_forecaster_results.json",
     "Minikube forecaster\n(16 hidden units, 1 layer)"),
    ("ml/models/lstm_forecaster_eks_results.json",
     "Amazon EKS forecaster\n(64 hidden units, 2 layers)"),
]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    for ax, (path, title) in zip(axes, PANELS):
        d = json.load(open(path))
        tr = d["history"]["train_loss"]
        va = d["history"]["val_loss"]
        epochs = range(1, len(tr) + 1)
        best = min(range(len(va)), key=lambda i: va[i])   # 0-indexed
        best_epoch = best + 1

        ax.plot(epochs, tr, color=TRAIN_C, lw=1.8, label="Training loss")
        ax.plot(epochs, va, color=VAL_C, lw=1.8, label="Validation loss")
        ax.axvline(best_epoch, color=MARK_C, ls="--", lw=1.2,
                   label=f"best epoch ({best_epoch})")
        ax.plot([best_epoch], [va[best]], "o", color=MARK_C, ms=6, zorder=5)
        ax.set_xlabel("Training epoch")
        ax.set_ylabel("Mean squared error (normalised units)")
        ax.set_title(
            f"{title}\nbest validation loss {va[best]:.4f} at epoch {best_epoch}; "
            f"stopped at epoch {len(tr)}",
            fontsize=9.5,
        )
        ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "LSTM training and validation loss, with early stopping 20 epochs "
        "after the best validation loss",
        fontweight="bold", fontsize=11,
    )
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.tight_layout()

    # PNG keeps DejaVu Sans so it matches the other thesis figures.
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")

    # SVG is re-emitted in Arial. DejaVu Sans is bundled with matplotlib and is
    # usually not installed as a system font, so an editor would substitute a
    # fallback and shift the layout. Arial resolves on macOS and Windows alike,
    # which keeps every label editable and correctly positioned.
    for t in fig.findobj(match=lambda o: hasattr(o, "set_fontfamily")):
        try:
            t.set_fontfamily("Arial")
        except Exception:
            pass
    fig.savefig(OUT_SVG, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_PNG}  (DejaVu Sans, matches other figures)")
    print(f"saved {OUT_SVG}  (Arial, editable text)")


if __name__ == "__main__":
    main()
