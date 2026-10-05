"""Plots and tabulates the LR study's runs - see LR_STUDY_PLAN.md.

    python -m benchmarks.lr_study.report workspace/lr_study/runs/p1_* \\
        --out workspace/lr_study/phase1.png
    python -m benchmarks.lr_study.report workspace/lr_study/runs/p2_* \\
        --out workspace/lr_study/phase2.png

Two kinds of run, told apart by their files:

LR range tests (lr_range_test.py's step_diagnostics.jsonl). Runs are grouped by name with
the trailing _s<seed> removed (one color per group, one line style per seed). For each run,
against the LR: the smoothed training loss (raw loss faintly behind it), the weight norm,
the effective step size LR / ||w||^2 (see the plan - with batch norm and no weight decay
this, not the nominal LR, is what the updates feel), the gradient norm, and the
mixed-precision loss scale. Printed per run: the LR where the smoothed loss is lowest,
where it is first clearly worse than that (+5%), and where it blows up (x1.5 or
non-finite), plus how far the sweep got before stopping. (Logs written before 2026-09-28
recorded Keras's running epoch mean as "loss", not the step's own loss.)

Training runs (supervised_policy_trainer.py's metadata.json), per epoch against the step:
validation and training loss, learning rate, weight norm, effective step size and loss
scale. Printed per run: its LR and warmup, whether it went non-finite, and the training
and validation loss at step 4,000 and at the end.
"""
import argparse
import json
import math
import os
import re

import numpy as np

WORSE = 1.05     # "clearly worse": smoothed loss this far above its minimum
BLOWN_UP = 1.5   # "blown up": this far above, or non-finite


# --- LR range tests ----------------------------------------------------------------------

def load(run_dir):
    with open(os.path.join(run_dir, "step_diagnostics.jsonl")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {k: np.array([np.nan if r.get(k) is None else r[k] for r in rows], float)
            for k in ("step", "lr", "loss", "weight_norm", "grad_norm", "loss_scale")}


def smooth(values, alpha=0.2):
    """Exponential moving average that carries non-finite values through as NaN. With a
    record every 10 steps, alpha 0.2 averages over roughly the last 50 steps - enough to
    tame batch noise without lagging far behind a fast sweep."""
    out, avg = np.full(len(values), np.nan), None
    for i, v in enumerate(values):
        if not math.isfinite(v):
            avg = None
            continue
        avg = v if avg is None else alpha * v + (1 - alpha) * avg
        out[i] = avg
    return out


def key_points(run):
    lr, loss = run["lr"], smooth(run["loss"])
    finite = np.isfinite(loss)
    i_min = int(np.nanargmin(loss))
    after = np.arange(len(loss)) > i_min
    worse = after & finite & (loss > loss[i_min] * WORSE)
    blown = after & (~np.isfinite(run["loss"]) | (finite & (loss > loss[i_min] * BLOWN_UP)))

    def first(mask):
        return float(lr[np.argmax(mask)]) if mask.any() else None
    return {"lr_at_min": float(lr[i_min]), "min_loss": float(loss[i_min]),
            "lr_worse": first(worse), "lr_blown": first(blown),
            "last_lr": float(lr[-1]), "last_step": int(run["step"][-1])}


def group_and_seed(run_dir):
    name = os.path.basename(os.path.normpath(run_dir))
    m = re.match(r"(.*)_s(\d+)$", name)
    return (m.group(1), m.group(2)) if m else (name, "")


def fmt(v):
    return "-" if v is None else "{:.3g}".format(v)


def range_report(dirs, out):
    runs = [(d, load(d)) for d in dirs]
    print("| run | lowest loss | at LR | +5% worse at LR | blown up at LR | sweep reached |")
    print("|---|---|---|---|---|---|")
    for d, run in runs:
        k = key_points(run)
        print("| {} | {:.3f} | {} | {} | {} | LR {} (step {}) |".format(
            os.path.basename(os.path.normpath(d)), k["min_loss"], fmt(k["lr_at_min"]),
            fmt(k["lr_worse"]), fmt(k["lr_blown"]), fmt(k["last_lr"]), k["last_step"]))
    if not out:
        return
    plt = _pyplot()
    groups = sorted({group_and_seed(d)[0] for d, _ in runs})
    colors = {g: plt.cm.tab10(i) for i, g in enumerate(groups)}
    seeds = sorted({group_and_seed(d)[1] for d, _ in runs})
    styles = {s: ["-", "--", ":", "-."][i % 4] for i, s in enumerate(seeds)}
    titles = ["training loss (smoothed)", "weight norm ||w||", "effective step  LR / ||w||^2",
              "gradient norm", "loss scale (mixed precision)"]
    fig, axes = plt.subplots(len(titles), 1, figsize=(11, 4 * len(titles)), sharex=True)
    for d, run in runs:
        g, s = group_and_seed(d)
        style = dict(color=colors[g], linestyle=styles[s], label="{} seed {}".format(g, s))
        lr = run["lr"]
        axes[0].plot(lr, run["loss"], color=colors[g], alpha=0.15, linewidth=0.8)
        axes[0].plot(lr, smooth(run["loss"]), **style)
        axes[1].plot(lr, run["weight_norm"], **style)
        axes[2].plot(lr, lr / run["weight_norm"] ** 2, **style)
        axes[3].plot(lr, run["grad_norm"], **style)
        axes[4].plot(lr, run["loss_scale"], **style)
    losses = np.concatenate([smooth(r["loss"]) for _, r in runs])
    losses = losses[np.isfinite(losses)]
    if len(losses):
        axes[0].set_ylim(losses.min() * 0.97, np.percentile(losses, 99) * 1.1)
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
    for ax in axes[2:]:
        ax.set_yscale("log")
    axes[-1].set_xlabel("learning rate")
    axes[0].legend(loc="upper left", fontsize=8)
    _save(fig, out)


# --- training runs -----------------------------------------------------------------------

def load_training_run(run_dir):
    """Per-epoch arrays from a training run's metadata.json, with the step each epoch
    ended at."""
    with open(os.path.join(run_dir, "metadata.json")) as f:
        meta = json.load(f)
    args = meta["cmd_line_args"][0]
    epochs = meta["epochs"]
    run = {k: np.array([np.nan if e.get(k) is None else e[k] for e in epochs], float)
           for k in ("loss", "val_loss", "learning_rate", "weight_norm", "loss_scale")}
    run["step"] = np.arange(1, len(epochs) + 1) * args["steps_per_epoch"]
    run["args"] = args
    return run


def _at_step(run, key, step):
    """The value at the last epoch ending at or before step (NaN if none)."""
    i = np.searchsorted(run["step"], step, side="right") - 1
    return float(run[key][i]) if i >= 0 else float("nan")


def training_report(dirs, out):
    runs = [(d, load_training_run(d)) for d in dirs]
    print("| run | LR | warmup | steps done | non-finite | loss @4000 | val @4000 | "
          "loss @end | val @end | min loss scale |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for d, run in runs:
        a = run["args"]
        bad = not (np.all(np.isfinite(run["loss"])) and np.all(np.isfinite(run["val_loss"])))
        end = int(run["step"][-1]) if len(run["step"]) else 0
        scale = run["loss_scale"]
        print("| {} | {} | {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {} |".format(
            os.path.basename(os.path.normpath(d)), fmt(a["learning_rate"]),
            a["warmup_steps"], end, "YES" if bad else "no",
            _at_step(run, "loss", 4000), _at_step(run, "val_loss", 4000),
            _at_step(run, "loss", end), _at_step(run, "val_loss", end),
            fmt(float(np.nanmin(scale))) if np.isfinite(scale).any() else "-"))
    if not out:
        return
    plt = _pyplot()
    lrs = sorted({r["args"]["learning_rate"] for _, r in runs})
    warmups = sorted({r["args"]["warmup_steps"] for _, r in runs})
    colors = {lr: plt.cm.tab10(i) for i, lr in enumerate(lrs)}
    styles = {w: ["-", "--", ":", "-."][i % 4] for i, w in enumerate(warmups)}
    panels = [("validation loss", "val_loss"), ("training loss (epoch mean)", "loss"),
              ("learning rate (used during the epoch)", "learning_rate"),
              ("weight norm ||w||", "weight_norm"), ("effective step  LR / ||w||^2", None),
              ("loss scale (mixed precision)", "loss_scale")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 3.6 * len(panels)), sharex=True)
    for d, run in runs:
        a = run["args"]
        label = "LR {} warmup {}".format(a["learning_rate"], a["warmup_steps"])
        style = dict(color=colors[a["learning_rate"]], linestyle=styles[a["warmup_steps"]],
                     marker=".", label=label)
        for ax, (_title, key) in zip(axes, panels):
            y = (run["learning_rate"] / run["weight_norm"] ** 2) if key is None else run[key]
            ax.plot(run["step"], y, **style)
    for ax, (title, _key) in zip(axes, panels):
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
    for ax in (axes[4], axes[5]):
        if any(np.any(np.isfinite(line.get_ydata()) & (line.get_ydata() > 0))
               for line in ax.get_lines()):
            ax.set_yscale("log")  # skipped when a panel has no data (older runs)
    finite = np.concatenate([r["val_loss"] for _, r in runs])
    finite = finite[np.isfinite(finite)]
    if len(finite):
        axes[0].set_ylim(finite.min() * 0.98, min(finite.max(), finite.min() * 1.6))
    axes[-1].set_xlabel("step")
    axes[0].legend(loc="upper right", fontsize=8)
    _save(fig, out)


# --- shared ------------------------------------------------------------------------------

def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, out):
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("saved", out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", help="Run directories, all of one kind")
    parser.add_argument("--out", help="Where to save the figure (PNG)")
    args = parser.parse_args(argv)
    range_runs = [d for d in args.runs
                  if os.path.exists(os.path.join(d, "step_diagnostics.jsonl"))]
    training_runs = [d for d in args.runs if d not in range_runs
                     and os.path.exists(os.path.join(d, "metadata.json"))]
    if range_runs and training_runs:
        parser.error("give range tests and training runs separately")
    if range_runs:
        range_report(range_runs, args.out)
    elif training_runs:
        training_report(training_runs, args.out)
    else:
        parser.error("no run with step_diagnostics.jsonl or metadata.json found")


if __name__ == "__main__":
    main()
