
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_utils import (
    STATE_ORDER,
    binary_metrics,
    calculate_confusion_bp,
    collapse_to_binary,
    ensure_pred_columns,
    ensure_truth_columns,
    load_tsv,
)


BASE_DIR = Path(".")
RUN_PREFIX = "2d.daiseg.seed"

SEED_START = 1
SEED_END = 50

MODERN_REF = 250
ND_REF = 3

OUT_DIR = Path("length_bin_analysis.ref250.nd3")

OUT_RECALL_PDF = OUT_DIR / "length_bin_recall.mean_across_runs.pdf"
OUT_PRECISION_PDF = OUT_DIR / "length_bin_precision.mean_across_runs.pdf"

LENGTH_BINS = [
    ("0_10kb", 0, 10_000),
    ("10_20kb", 10_000, 20_000),
    ("20_25kb", 20_000, 25_000),
    ("25_30kb", 25_000, 30_000),
    ("30_40kb", 30_000, 40_000),
    ("40_50kb", 40_000, 50_000),
    ("50_70kb", 50_000, 70_000),
    ("70_100kb", 70_000, 100_000),
    ("100kb_plus", 100_000, None),
]


def add_length_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Length"] = out["End"] - out["Start"]
    return out


def filter_truth_bin(
    df: pd.DataFrame,
    start_bp: int,
    end_bp: int | None,
) -> pd.DataFrame:
    if end_bp is None:
        return df[df["Length"] >= start_bp].copy()

    return df[
        (df["Length"] >= start_bp)
        & (df["Length"] < end_bp)
    ].copy()


def row_normalize_nan(conf: np.ndarray) -> np.ndarray:
    """Row-normalize while leaving undefined rows as NaN."""
    row_sums = conf.sum(axis=1, keepdims=True)
    return np.divide(
        conf,
        row_sums,
        out=np.full(conf.shape, np.nan, dtype=float),
        where=row_sums != 0,
    )


def column_normalize_nan(conf: np.ndarray) -> np.ndarray:
    """Column-normalize while leaving undefined columns as NaN."""
    col_sums = conf.sum(axis=0, keepdims=True)
    return np.divide(
        conf,
        col_sums,
        out=np.full(conf.shape, np.nan, dtype=float),
        where=col_sums != 0,
    )


def format_bin_title(
    bin_name: str,
    recall: float,
    precision: float,
) -> str:
    return (
        f"{bin_name}\n"
        f"R={recall:.3f}, P={precision:.3f}"
    )


def plot_results(
    matrices: dict,
    bin_summary: dict,
    out_pdf: Path,
    matrix_label: str,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))

    for ax, (bin_name, _, _) in zip(
        axes.flat,
        LENGTH_BINS,
    ):
        mat = matrices[bin_name]
        stats = bin_summary[bin_name]

        ax.imshow(
            mat,
            cmap="OrRd",
            vmin=0,
            vmax=1,
        )

        ax.set_xticks(range(len(STATE_ORDER)))
        ax.set_xticklabels(
            STATE_ORDER,
            rotation=45,
            ha="right",
        )

        ax.set_yticks(range(len(STATE_ORDER)))
        ax.set_yticklabels(STATE_ORDER)

        ax.set_title(
            format_bin_title(
                bin_name,
                stats["binary_recall"],
                stats["binary_precision"],
            ),
            fontsize=9,
        )

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if np.isnan(val):
                    label = "NA"
                    color = "black"
                else:
                    label = f"{val:.3f}"
                    color = "white" if val > 0.5 else "black"

                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )

    fig.suptitle(
        (
            f"Length-stratified {matrix_label} "
            f"(mean across runs; ref={MODERN_REF}, nd={ND_REF})"
        ),
        y=0.98,
    )

    fig.tight_layout()
    fig.savefig(
        out_pdf,
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Each run is normalized separately. Undefined rows or columns are
    # represented by NaN and therefore excluded from the across-run mean.
    bin_conf5_raw = {
        name: []
        for name, _, _ in LENGTH_BINS
    }

    bin_binary_metrics = {
        name: []
        for name, _, _ in LENGTH_BINS
    }

    completed_runs = 0

    for run in range(
        SEED_START,
        SEED_END + 1,
    ):
        run_dir = BASE_DIR / f"{RUN_PREFIX}{run}"

        truth_path = (
            run_dir
            / "raw"
            / "truth.all.tsv"
        )

        pred_path = (
            run_dir
            / "runs"
            / "daiseg_mexicans"
            / (
                f"ref.eu{MODERN_REF}."
                f"na{MODERN_REF}."
                f"af{MODERN_REF}."
                f"nd{ND_REF}"
            )
            / "all.inferred.daiseg_mexicans.em.tsv"
        )

        if not truth_path.exists():
            print(f"[skip] missing truth: {truth_path}")
            continue

        if not pred_path.exists():
            print(f"[skip] missing prediction: {pred_path}")
            continue

        truth = ensure_truth_columns(
            load_tsv(truth_path)
        )
        truth = add_length_column(truth)

        pred = ensure_pred_columns(
            load_tsv(pred_path)
        )

        for bin_name, start_bp, end_bp in LENGTH_BINS:
            gt_bin = filter_truth_bin(
                truth,
                start_bp,
                end_bp,
            )

            conf5 = calculate_confusion_bp(
                gt_bin,
                pred,
            )
            conf2 = collapse_to_binary(conf5)

            bin_conf5_raw[bin_name].append(conf5)
            bin_binary_metrics[bin_name].append(
                binary_metrics(conf2)
            )

        completed_runs += 1
        print(f"[ok] run={run}")

    if completed_runs == 0:
        raise SystemExit("No completed runs found.")

    summary = {}
    recall_matrices = {}
    precision_matrices = {}

    for bin_name, _, _ in LENGTH_BINS:
        recall_stack = np.stack(
            [
                row_normalize_nan(conf)
                for conf in bin_conf5_raw[bin_name]
            ],
            axis=0,
        )

        precision_stack = np.stack(
            [
                column_normalize_nan(conf)
                for conf in bin_conf5_raw[bin_name]
            ],
            axis=0,
        )

        recalls = [
            metrics["recall"]
            for metrics in bin_binary_metrics[bin_name]
        ]
        precisions = [
            metrics["precision"]
            for metrics in bin_binary_metrics[bin_name]
        ]

        summary[bin_name] = {
            "binary_recall": float(
                np.nanmean(recalls)
            ),
            "binary_precision": float(
                np.nanmean(precisions)
            ),
        }

        # Rows are true states and columns are predicted states.
        # Row-normalized diagonal entries are state-specific recall.
        recall_matrices[bin_name] = np.nanmean(
            recall_stack,
            axis=0,
        )

        # Column-normalized diagonal entries are state-specific precision.
        precision_matrices[bin_name] = np.nanmean(
            precision_stack,
            axis=0,
        )

    # Save numerical source data underlying the length-bin figures.
    states = ["EU", "ND_EU", "NA", "ND_NA", "AF"]

    source_rows = []

    for bin_name, start_bp, end_bp in LENGTH_BINS:
        # Binary summary shown with the figures.
        source_rows.append({
            "length_bin": bin_name,
            "start_bp": start_bp,
            "end_bp": end_bp,
            "matrix_type": "binary_summary",
            "true_state": "",
            "predicted_state": "",
            "metric": "archaic_recall",
            "value": summary[bin_name]["binary_recall"],
        })

        source_rows.append({
            "length_bin": bin_name,
            "start_bp": start_bp,
            "end_bp": end_bp,
            "matrix_type": "binary_summary",
            "true_state": "",
            "predicted_state": "",
            "metric": "archaic_precision",
            "value": summary[bin_name]["binary_precision"],
        })

        # Row-normalized matrix used in the recall figure.
        mat = recall_matrices[bin_name]
        for i, true_state in enumerate(states):
            for j, pred_state in enumerate(states):
                source_rows.append({
                    "length_bin": bin_name,
                    "start_bp": start_bp,
                    "end_bp": end_bp,
                    "matrix_type": "recall_matrix",
                    "true_state": true_state,
                    "predicted_state": pred_state,
                    "metric": "value",
                    "value": mat[i, j],
                })

        # Column-normalized matrix used in the precision figure.
        mat = precision_matrices[bin_name]
        for i, true_state in enumerate(states):
            for j, pred_state in enumerate(states):
                source_rows.append({
                    "length_bin": bin_name,
                    "start_bp": start_bp,
                    "end_bp": end_bp,
                    "matrix_type": "precision_matrix",
                    "true_state": true_state,
                    "predicted_state": pred_state,
                    "metric": "value",
                    "value": mat[i, j],
                })

    source_path = Path(__file__).resolve().parent / \
        "S6_Data_length_bin_performance.tsv"

    pd.DataFrame(source_rows).to_csv(
        source_path,
        sep="\t",
        index=False,
    )

    print(f"Saved source data to {source_path}")

    plot_results(
        matrices=recall_matrices,
        bin_summary=summary,
        out_pdf=OUT_RECALL_PDF,
        matrix_label="row-normalized confusion matrices",
    )

    plot_results(
        matrices=precision_matrices,
        bin_summary=summary,
        out_pdf=OUT_PRECISION_PDF,
        matrix_label="precision matrices",
    )

    print(f"Saved recall plot to {OUT_RECALL_PDF}")
    print(f"Saved precision plot to {OUT_PRECISION_PDF}")


if __name__ == "__main__":
    main()
