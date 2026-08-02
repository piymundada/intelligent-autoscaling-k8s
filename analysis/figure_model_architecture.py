#!/usr/bin/env python3
"""Figure 4.5.1: how a window of metrics becomes a replica count.

Derived directly from ml/lstm_model.py and ml/autoscaler.py rather than drawn by hand, so the
tensor shapes, layer names and parameter counts in the figure are the ones the code produces.
Parameter counts are computed from the PyTorch LSTM gate formula and cross-checked against
torch.nn.LSTM when torch is importable.

House style matched to the hand-drawn SVGs in docs/. Writes PNG and SVG; re-running overwrites
both, so save a hand-edited SVG under a different name first.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.lines import Line2D

INK, SECONDARY, CONNECTOR = "#1f1f1f", "#5F5E5A", "#7a7a74"
BLUE = ("#378ADD", "#E6F1FB")     # the network itself
GREEN = ("#639922", "#EAF3DE")    # what the controller does with the output
AMBER = ("#BA7517", "#FAEEDA")    # input

plt.rcParams["font.family"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"

SEQ_LEN, N_FEATURES = 10, 10
LOCAL = dict(hidden=16, layers=1)
CLOUD = dict(hidden=64, layers=2)


def lstm_params(input_size, hidden, layers):
    """PyTorch nn.LSTM parameter count: 4 gates, two bias vectors per layer."""
    total = 0
    for layer in range(layers):
        in_dim = input_size if layer == 0 else hidden
        total += 4 * hidden * in_dim + 4 * hidden * hidden + 4 * hidden + 4 * hidden
    return total


def totals(cfg):
    rec = lstm_params(N_FEATURES, cfg["hidden"], cfg["layers"])
    out = cfg["hidden"] + 1
    return rec, out, rec + out


# (title, shape/detail, source line, accent)
STEPS = [
    ("Normalised metric window",
     f"({SEQ_LEN} time steps × {N_FEATURES} features), each value scaled to 0 to 1",
     "five minutes of history at 30 s sampling", AMBER),
    ("nn.LSTM(batch_first=True)",
     "1 layer × 16 units locally, 2 layers × 64 units on EKS",
     "reads the window one step at a time, carrying state forward", BLUE),
    ("Recurrent output, all steps",
     f"(batch, {SEQ_LEN}, hidden)",
     "one hidden vector per input time step", BLUE),
    ("lstm_out[:, -1, :]",
     "(batch, hidden)",
     "keep only the final step: the network's summary of the whole window", BLUE),
    ("nn.Dropout(0.2)",
     "(batch, hidden)",
     "active in training only, disabled at inference", BLUE),
    ("nn.Linear(hidden, 1) then squeeze(-1)",
     "(batch,)",
     "one number per window", BLUE),
    ("Normalised request-rate forecast",
     "a single value in 0 to 1, 60 s ahead",
     "the model's output ends here", BLUE),
    ("Denormalise with the saved scaler",
     "RPS = value × (max − min) + min",
     "converts the forecast to requests per second", GREEN),
    ("Replica count",
     "ceil(RPS ÷ sustainable RPS per replica × 1.5), clamped",
     "the controller's forecast-based recommendation, before the JVM floor", GREEN),
]

X0, X1 = 2.0, 98.0
BOX_H, GAP = 8.6, 3.4
ROUND = 1.1


def draw(outfile="docs/figure_4_5_1_model_dataflow"):
    H = len(STEPS) * (BOX_H + GAP) + 2.0
    fig_w = 10.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * (H / (X1 - X0)) * 0.92))
    ax.set_xlim(0, 100)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    y = 1.0
    for i, (title, shape, note, (accent, pastel)) in enumerate(STEPS):
        ax.add_patch(FancyBboxPatch(
            (X0, y), X1 - X0, BOX_H,
            boxstyle=f"round,pad=0,rounding_size={ROUND}",
            linewidth=1.5, edgecolor=accent, facecolor=pastel,
        ))
        ax.text(X0 + 3.0, y + 2.5, title, ha="left", va="center",
                fontsize=11.5, fontweight="semibold", color=INK)
        ax.text(X0 + 3.0, y + 5.3, shape, ha="left", va="center",
                fontsize=10, color=INK)
        ax.text(X0 + 3.0, y + 7.4, note, ha="left", va="center",
                fontsize=9.2, color=SECONDARY, style="italic")
        y += BOX_H

        if i < len(STEPS) - 1:
            top, bot = y + 0.7, y + GAP - 0.7
            ax.add_line(Line2D([50, 50], [top, bot - 1.1], color=CONNECTOR,
                               linewidth=1.5, solid_capstyle="butt"))
            ax.add_patch(Polygon([[48.4, bot - 1.2], [51.6, bot - 1.2], [50, bot]],
                                 closed=True, facecolor=CONNECTOR, linewidth=0))
            y += GAP

    fig.tight_layout(pad=0.25)
    fig.savefig(f"{outfile}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{outfile}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {outfile}.png and .svg")


if __name__ == "__main__":
    for name, cfg in (("Minikube", LOCAL), ("Amazon EKS", CLOUD)):
        rec, out, tot = totals(cfg)
        print(f"{name:11} hidden={cfg['hidden']:>2} layers={cfg['layers']}  "
              f"recurrent={rec:>6,}  output={out:>3,}  total={tot:>6,}")
    try:
        import torch.nn as nn
        for name, cfg in (("Minikube", LOCAL), ("Amazon EKS", CLOUD)):
            m = nn.LSTM(N_FEATURES, cfg["hidden"], cfg["layers"], batch_first=True)
            n = sum(p.numel() for p in m.parameters())
            assert n == lstm_params(N_FEATURES, cfg["hidden"], cfg["layers"]), name
        print("cross-checked against torch.nn.LSTM: OK")
    except ImportError:
        print("torch not importable here; counts come from the gate formula")
    draw()
