"""Minimal transformer demo for multiplexed gas-sensor data.

Run from this folder:
    python run_demo.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


DEFAULT_GAS_NAMES = ["H2", "CO", "CO2", "NO2"]


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 8192) -> None:
        super().__init__()
        self.dropout = nn.Dropout(float(dropout))
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :].to(dtype=x.dtype, device=x.device))


class GasTransformer(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        gas_dim: int,
        embed_dim: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        attn_out_dim: int,
        max_seq_len: int,
        input_layernorm: bool,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim) if input_layernorm else nn.Identity()
        self.in_proj = nn.Linear(feature_dim, embed_dim)
        self.pos = SinusoidalPositionalEncoding(embed_dim, dropout=dropout, max_len=max_seq_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=max(256, 4 * embed_dim),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pred_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, gas_dim))
        self.attn_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, attn_out_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_norm(x)
        h = self.in_proj(h)
        h = self.pos(h)
        h = self.encoder(h)
        return torch.sigmoid(self.pred_head(h)), self.attn_head(h)


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[GasTransformer, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    feature_dim = int(state["in_proj.weight"].shape[1])
    embed_dim = int(state["in_proj.weight"].shape[0])
    gas_dim = int(state["pred_head.1.weight"].shape[0])
    attn_out_dim = int(state["attn_head.1.weight"].shape[0])
    seq_len = int(cfg.get("seq_len", 256))

    model = GasTransformer(
        feature_dim=feature_dim,
        gas_dim=gas_dim,
        embed_dim=int(cfg.get("embed_dim", embed_dim)),
        nhead=int(cfg.get("nhead", 8)),
        num_layers=int(cfg.get("num_layers", 4)),
        dropout=float(cfg.get("dropout", 0.0)),
        attn_out_dim=attn_out_dim,
        max_seq_len=max(512, seq_len),
        input_layernorm=bool(cfg.get("input_layernorm", True)),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg


def infer_full_sequence(
    model: GasTransformer,
    X: np.ndarray,
    *,
    seq_len: int,
    stride: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    T = int(X.shape[0])
    if T < seq_len:
        pad = seq_len - T
        mode = "reflect" if T > 1 else "edge"
        pred, attn = infer_full_sequence(
            model,
            np.pad(X, ((0, pad), (0, 0)), mode=mode),
            seq_len=seq_len,
            stride=seq_len,
            batch_size=batch_size,
            device=device,
        )
        return pred[:T], attn[:T]

    starts = list(range(0, T - seq_len + 1, max(1, int(stride))))
    if starts[-1] != T - seq_len:
        starts.append(T - seq_len)

    with torch.no_grad():
        probe = torch.from_numpy(X[starts[0] : starts[0] + seq_len][None, ...]).to(device)
        pred_probe, attn_probe = model(probe)
    gas_dim = int(pred_probe.shape[-1])
    feature_dim = int(attn_probe.shape[-1])

    sum_pred = np.zeros((T, gas_dim), dtype=np.float32)
    sum_attn = np.zeros((T, feature_dim), dtype=np.float32)
    counts = np.zeros((T,), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, len(starts), max(1, int(batch_size))):
            batch_starts = starts[i : i + max(1, int(batch_size))]
            batch = np.stack([X[s : s + seq_len] for s in batch_starts], axis=0)
            pred_b, attn_logits_b = model(torch.from_numpy(batch).to(device))
            pred_np = pred_b.cpu().numpy().astype(np.float32, copy=False)
            attn_np = torch.softmax(attn_logits_b, dim=-1).cpu().numpy().astype(np.float32, copy=False)
            for j, s in enumerate(batch_starts):
                e = s + seq_len
                sum_pred[s:e] += pred_np[j]
                sum_attn[s:e] += attn_np[j]
                counts[s:e] += 1.0

    counts = np.maximum(counts, 1.0)[:, None]
    return sum_pred / counts, sum_attn / counts


def plot_results(
    *,
    time_hours: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    attention: np.ndarray,
    output_path: Path,
    title: str,
    gas_names: list[str],
    max_points: int,
) -> None:
    n = time_hours.shape[0]
    show = np.unique(np.linspace(0, n - 1, max_points, dtype=np.int64)) if n > max_points else np.arange(n)

    t = time_hours[show]
    pred = prediction[show]
    y = target[show]
    attn = attention[show]

    colors = ["#163b7c", "#0b8a8f", "#d1982d", "#c45a34"]
    fig, axes = plt.subplots(
        prediction.shape[1] + 1,
        1,
        figsize=(12.0, 8.0),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 1.25][: prediction.shape[1] + 1]},
    )
    fig.patch.set_facecolor("#f5f7fb")

    for i in range(prediction.shape[1]):
        ax = axes[i]
        ax.set_facecolor("white")
        name = gas_names[i] if i < len(gas_names) else f"Gas {i + 1}"
        ax.plot(t, y[:, i], color="#101828", lw=1.25, ls="--", label="target" if i == 0 else None)
        ax.plot(t, pred[:, i], color=colors[i % len(colors)], lw=1.35, label="model prediction" if i == 0 else None)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, color="#dce5ef", lw=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].legend(loc="upper right", frameon=True)

    ax = axes[-1]
    ax.set_facecolor("white")
    vmax = float(np.quantile(attn, 0.995)) if attn.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    extent = [float(t[0]), float(t[-1]), 1, attn.shape[1]] if t.size else [0, 1, 1, attn.shape[1]]
    im = ax.imshow(attn.T, aspect="auto", origin="lower", extent=extent, cmap="inferno", vmin=0.0, vmax=vmax)
    ax.set_ylabel("pixel", fontweight="bold")
    ax.set_xlabel("Time (h)")
    ax.set_title("Attention over 64 optical readout pixels", loc="left", fontsize=10, fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
    cbar.set_label("weight")

    fig.suptitle(f"Gas transformer demo: {title}", x=0.02, y=0.995, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run the multiplexed gas-transformer demo.")
    parser.add_argument("--data", type=Path, default=root / "Preprocessed Data" / "jakarta_demo_sequence.npz")
    parser.add_argument("--checkpoint", type=Path, default=root / "model_checkpoint.pt")
    parser.add_argument("--output", type=Path, default=root / "demo_predictions.png")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-points", type=int, default=6000)
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.checkpoint, device)

    data = np.load(args.data, allow_pickle=False)
    X = data["X"].astype(np.float32, copy=False)
    y = data["y"].astype(np.float32, copy=False)
    plot_mask = data["plot_mask"].astype(bool, copy=False)
    time_hours = data["plot_time_hours"].astype(np.float32, copy=False)
    label = str(data["label"])
    gas_names = [str(x) for x in data["gas_names"]] if "gas_names" in data.files else DEFAULT_GAS_NAMES

    pred, attn = infer_full_sequence(
        model,
        X,
        seq_len=int(cfg.get("seq_len", 256)),
        stride=int(cfg.get("stride", 32)),
        batch_size=int(args.batch_size),
        device=device,
    )

    plot_results(
        time_hours=time_hours,
        prediction=pred[plot_mask],
        target=y[plot_mask, : pred.shape[1]],
        attention=attn[plot_mask],
        output_path=args.output,
        title=label,
        gas_names=gas_names,
        max_points=int(args.max_points),
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
