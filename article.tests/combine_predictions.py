#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd


DEFAULT_BASE_DIR = "."


def get_project_dirs(base_dir: str, sim_name: str) -> Dict[str, Path]:
    root = Path(base_dir) / sim_name
    return {
        "root": root,
        "raw": root / "raw",
        "prepared": root / "prepared",
        "runs": root / "runs",
        "metrics": root / "metrics",
    }


def get_rfmix_run_dir(base_dir: str, sim_name: str, n_eu_ref: int, n_na_ref: int, n_af_ref: int) -> Path:
    dirs = get_project_dirs(base_dir, sim_name)
    return dirs["runs"] / "rfmix" / f"ref.eu{n_eu_ref}.na{n_na_ref}.af{n_af_ref}"


def get_hmmix_run_dir(base_dir: str, sim_name: str, n_af_ref: int) -> Path:
    dirs = get_project_dirs(base_dir, sim_name)
    return dirs["runs"] / "hmmix" / f"ref.af{n_af_ref}"


def get_daiseg_simple_run_dir(base_dir: str, sim_name: str, n_af_ref: int, n_nd_ref: int) -> Path:
    dirs = get_project_dirs(base_dir, sim_name)
    return dirs["runs"] / "daiseg_simple" / f"ref.af{n_af_ref}.nd{n_nd_ref}"


def get_combine_run_dir(base_dir: str, sim_name: str, mode: str, tag: str) -> Path:
    dirs = get_project_dirs(base_dir, sim_name)
    return dirs["runs"] / mode / tag


def load_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def load_hmmix_threshold_file(run_dir: Path, n_af_ref: int, threshold: float) -> Path:
    suffix = f"ref.{n_af_ref}"
    thr_str = f"{threshold:.2f}".replace(".", "_")
    path = run_dir / f"all.inferred.hmmix.{suffix}.thr{thr_str}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"HMMix threshold file not found: {path}")
    return path


def load_hmmix_viterbi_file(run_dir: Path, n_af_ref: int) -> Path:
    suffix = f"ref.{n_af_ref}"
    path = run_dir / f"all.inferred.hmmix.{suffix}.viterbi.tsv"
    if not path.exists():
        raise FileNotFoundError(f"HMMix viterbi file not found: {path}")
    return path


def normalize_rfmix_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = {"CHROM", "Sample", "Start", "End", "Length", "State"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"RFMix file is missing columns: {missing}")
    out["CHROM"] = out["CHROM"].astype(int)
    out["Sample"] = out["Sample"].astype(str)
    out["Start"] = out["Start"].astype(int)
    out["End"] = out["End"].astype(int)
    out["Length"] = out["Length"].astype(int)
    out["State"] = out["State"].astype(str)
    return out


def normalize_hmmix_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "CHROM" in out.columns and "CHR" not in out.columns:
        out["CHR"] = out["CHROM"]
    required = {"CHR", "Sample", "Start", "End"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"HMMix file is missing columns: {missing}")
    out["CHR"] = out["CHR"].astype(int)
    out["Sample"] = out["Sample"].astype(str).apply(lambda x: x if x.startswith("MX_") else f"MX_{x}")
    out["Start"] = out["Start"].astype(int)
    out["End"] = out["End"].astype(int)
    if "Length" not in out.columns:
        out["Length"] = out["End"] - out["Start"]
    out["Length"] = out["Length"].astype(int)
    return out[["CHR", "Sample", "Start", "End", "Length"] + ([c for c in ["mean_prob"] if c in out.columns])]


def normalize_daiseg_simple_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "CHROM" in out.columns and "CHR" not in out.columns:
        out["CHR"] = out["CHROM"]
    required = {"CHR", "Sample", "Start", "End"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"DAIseg.simple file is missing columns: {missing}")
    out["CHR"] = out["CHR"].astype(int)
    out["Sample"] = out["Sample"].astype(str).apply(lambda x: x if x.startswith("MX_") else f"MX_{x}")
    out["Start"] = out["Start"].astype(int)
    out["End"] = out["End"].astype(int)
    if "Length" not in out.columns:
        out["Length"] = out["End"] - out["Start"]
    out["Length"] = out["Length"].astype(int)
    return out[["CHR", "Sample", "Start", "End", "Length"]]


def combine_rfmix_hmmix(rfmix_df: pd.DataFrame, hmmix_df: pd.DataFrame) -> pd.DataFrame:
    if rfmix_df.empty:
        return pd.DataFrame(columns=["CHROM", "Sample", "Start", "End", "Length", "State"])
    if hmmix_df.empty:
        return rfmix_df.copy()

    rf = normalize_rfmix_df(rfmix_df)
    hm = normalize_hmmix_df(hmmix_df)

    out_rows = []
    for (sample, chrom), rf_grp in rf.groupby(["Sample", "CHROM"]):
        hm_sub = hm[(hm["Sample"] == sample) & (hm["CHR"] == chrom)].sort_values("Start")
        for _, r_seg in rf_grp.iterrows():
            r_s = int(r_seg["Start"])
            r_e = int(r_seg["End"])
            r_st = str(r_seg["State"])

            if hm_sub.empty or r_st == "AF":
                out_rows.append(r_seg.to_dict())
                continue

            overlaps = hm_sub[(hm_sub["Start"] < r_e) & (hm_sub["End"] > r_s)]
            if overlaps.empty:
                out_rows.append(r_seg.to_dict())
                continue

            cur = r_s
            for _, nd in overlaps.iterrows():
                ns = max(r_s, int(nd["Start"]))
                ne = min(r_e, int(nd["End"]))
                if ns >= ne:
                    continue
                if cur < ns:
                    out_rows.append({**r_seg.to_dict(), "Start": cur, "End": ns, "Length": ns - cur})
                nd_state = f"ND_{r_st}" if r_st in {"EU", "NA"} else r_st
                out_rows.append({**r_seg.to_dict(), "Start": ns, "End": ne, "Length": ne - ns, "State": nd_state})
                cur = max(cur, ne)
            if cur < r_e:
                out_rows.append({**r_seg.to_dict(), "Start": cur, "End": r_e, "Length": r_e - cur})

    res = pd.DataFrame(out_rows)
    if res.empty:
        return pd.DataFrame(columns=["CHROM", "Sample", "Start", "End", "Length", "State"])
    return res.sort_values(["CHROM", "Sample", "Start", "End"]).reset_index(drop=True)


def combine_rfmix_daiseg_simple(rfmix_df: pd.DataFrame, simple_df: pd.DataFrame) -> pd.DataFrame:
    if rfmix_df.empty:
        return pd.DataFrame(columns=["CHROM", "Sample", "Start", "End", "Length", "State"])
    if simple_df.empty:
        return rfmix_df.copy()

    rf = normalize_rfmix_df(rfmix_df)
    ds = normalize_daiseg_simple_df(simple_df)

    out_rows = []
    for (sample, chrom), rf_grp in rf.groupby(["Sample", "CHROM"]):
        ds_sub = ds[(ds["Sample"] == sample) & (ds["CHR"] == chrom)].sort_values("Start")
        for _, r_seg in rf_grp.iterrows():
            r_s = int(r_seg["Start"])
            r_e = int(r_seg["End"])
            r_st = str(r_seg["State"])

            if ds_sub.empty or r_st == "AF":
                out_rows.append(r_seg.to_dict())
                continue

            overlaps = ds_sub[(ds_sub["Start"] < r_e) & (ds_sub["End"] > r_s)]
            if overlaps.empty:
                out_rows.append(r_seg.to_dict())
                continue

            cur = r_s
            for _, nd in overlaps.iterrows():
                ns = max(r_s, int(nd["Start"]))
                ne = min(r_e, int(nd["End"]))
                if ns >= ne:
                    continue
                if cur < ns:
                    out_rows.append({**r_seg.to_dict(), "Start": cur, "End": ns, "Length": ns - cur})
                nd_state = f"ND_{r_st}" if r_st in {"EU", "NA"} else r_st
                out_rows.append({**r_seg.to_dict(), "Start": ns, "End": ne, "Length": ne - ns, "State": nd_state})
                cur = max(cur, ne)
            if cur < r_e:
                out_rows.append({**r_seg.to_dict(), "Start": cur, "End": r_e, "Length": r_e - cur})

    res = pd.DataFrame(out_rows)
    if res.empty:
        return pd.DataFrame(columns=["CHROM", "Sample", "Start", "End", "Length", "State"])
    return res.sort_values(["CHROM", "Sample", "Start", "End"]).reset_index(drop=True)


def combine_mode_rfmix_hmmix(
    sim_name: str,
    *,
    base_dir: str,
    rfmix_eu_ref: int,
    rfmix_na_ref: int,
    rfmix_af_ref: int,
    hmmix_af_ref: int,
    hmmix_threshold: float | None,
    viterbi: bool,
    force: bool,
) -> Path:
    rfmix_dir = get_rfmix_run_dir(base_dir, sim_name, rfmix_eu_ref, rfmix_na_ref, rfmix_af_ref)
    rfmix_path = rfmix_dir / "rfmix.all.tsv"
    rfmix_df = load_tsv(rfmix_path)

    hmmix_dir = get_hmmix_run_dir(base_dir, sim_name, hmmix_af_ref)
    if viterbi:
        hmmix_path = load_hmmix_viterbi_file(hmmix_dir, hmmix_af_ref)
        tag = f"rfmix.eu{rfmix_eu_ref}.na{rfmix_na_ref}.af{rfmix_af_ref}__hmmix.af{hmmix_af_ref}.viterbi"
    else:
        if hmmix_threshold is None:
            raise ValueError("--hmmix-threshold is required unless --viterbi is used")
        hmmix_path = load_hmmix_threshold_file(hmmix_dir, hmmix_af_ref, hmmix_threshold)
        thr_tag = f"{hmmix_threshold:.2f}".replace(".", "_")
        tag = f"rfmix.eu{rfmix_eu_ref}.na{rfmix_na_ref}.af{rfmix_af_ref}__hmmix.af{hmmix_af_ref}.thr{thr_tag}"

    hmmix_df = load_tsv(hmmix_path)
    combined_df = combine_rfmix_hmmix(rfmix_df, hmmix_df)

    out_dir = get_combine_run_dir(base_dir, sim_name, "rfmix_hmmix", tag)
    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "combined.predictions.tsv"
    combined_df.to_csv(out_file, sep="\t", index=False)

    manifest = {
        "mode": "rfmix_hmmix",
        "sim_name": sim_name,
        "rfmix": {
            "run_dir": str(rfmix_dir.resolve()),
            "n_eu_ref": int(rfmix_eu_ref),
            "n_na_ref": int(rfmix_na_ref),
            "n_af_ref": int(rfmix_af_ref),
            "input_file": rfmix_path.name,
        },
        "hmmix": {
            "run_dir": str(hmmix_dir.resolve()),
            "n_af_ref": int(hmmix_af_ref),
            "viterbi": bool(viterbi),
            "threshold": None if viterbi else float(hmmix_threshold),
            "input_file": hmmix_path.name,
        },
        "output_file": out_file.name,
    }
    with open(out_dir / "manifest.combine.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 72)
    print("Combine completed: rfmix_hmmix")
    print(f"Output dir:   {out_dir}")
    print(f"Output file:  {out_file}")
    print(f"Rows:         {len(combined_df)}")
    print("=" * 72)
    return out_dir


def combine_mode_rfmix_daiseg_simple(
    sim_name: str,
    *,
    base_dir: str,
    rfmix_eu_ref: int,
    rfmix_na_ref: int,
    rfmix_af_ref: int,
    simple_af_ref: int,
    simple_nd_ref: int,
    force: bool,
) -> Path:
    rfmix_dir = get_rfmix_run_dir(base_dir, sim_name, rfmix_eu_ref, rfmix_na_ref, rfmix_af_ref)
    rfmix_path = rfmix_dir / "rfmix.all.tsv"
    rfmix_df = load_tsv(rfmix_path)

    simple_dir = get_daiseg_simple_run_dir(base_dir, sim_name, simple_af_ref, simple_nd_ref)
    simple_path = simple_dir / "all.inferred.daiseg_simple.em_v2.tsv"
    simple_df = load_tsv(simple_path)

    combined_df = combine_rfmix_daiseg_simple(rfmix_df, simple_df)
    tag = f"rfmix.eu{rfmix_eu_ref}.na{rfmix_na_ref}.af{rfmix_af_ref}__simple.af{simple_af_ref}.nd{simple_nd_ref}"

    out_dir = get_combine_run_dir(base_dir, sim_name, "rfmix_daiseg_simple", tag)
    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "combined.predictions.tsv"
    combined_df.to_csv(out_file, sep="\t", index=False)

    manifest = {
        "mode": "rfmix_daiseg_simple",
        "sim_name": sim_name,
        "rfmix": {
            "run_dir": str(rfmix_dir.resolve()),
            "n_eu_ref": int(rfmix_eu_ref),
            "n_na_ref": int(rfmix_na_ref),
            "n_af_ref": int(rfmix_af_ref),
            "input_file": rfmix_path.name,
        },
        "daiseg_simple": {
            "run_dir": str(simple_dir.resolve()),
            "n_af_ref": int(simple_af_ref),
            "n_nd_ref": int(simple_nd_ref),
            "input_file": simple_path.name,
        },
        "output_file": out_file.name,
    }
    with open(out_dir / "manifest.combine.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 72)
    print("Combine completed: rfmix_daiseg_simple")
    print(f"Output dir:   {out_dir}")
    print(f"Output file:  {out_file}")
    print(f"Rows:         {len(combined_df)}")
    print("=" * 72)
    return out_dir


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine mex_compare predictions into 5-state calls")
    sub = parser.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("rfmix_hmmix")
    p1.add_argument("--sim-name", type=str, required=True)
    p1.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    p1.add_argument("--rfmix-eu-ref", type=int, required=True)
    p1.add_argument("--rfmix-na-ref", type=int, required=True)
    p1.add_argument("--rfmix-af-ref", type=int, required=True)
    p1.add_argument("--hmmix-af-ref", type=int, required=True)
    p1.add_argument("--hmmix-threshold", type=float, default=None)
    p1.add_argument("--viterbi", action="store_true")
    p1.add_argument("--force", action="store_true")

    p2 = sub.add_parser("rfmix_daiseg_simple")
    p2.add_argument("--sim-name", type=str, required=True)
    p2.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    p2.add_argument("--rfmix-eu-ref", type=int, required=True)
    p2.add_argument("--rfmix-na-ref", type=int, required=True)
    p2.add_argument("--rfmix-af-ref", type=int, required=True)
    p2.add_argument("--simple-af-ref", type=int, required=True)
    p2.add_argument("--simple-nd-ref", type=int, required=True)
    p2.add_argument("--force", action="store_true")

    return parser


def main() -> None:
    args = make_parser().parse_args()

    if args.mode == "rfmix_hmmix":
        combine_mode_rfmix_hmmix(
            sim_name=args.sim_name,
            base_dir=args.base_dir,
            rfmix_eu_ref=args.rfmix_eu_ref,
            rfmix_na_ref=args.rfmix_na_ref,
            rfmix_af_ref=args.rfmix_af_ref,
            hmmix_af_ref=args.hmmix_af_ref,
            hmmix_threshold=args.hmmix_threshold,
            viterbi=args.viterbi,
            force=args.force,
        )
        return

    if args.mode == "rfmix_daiseg_simple":
        combine_mode_rfmix_daiseg_simple(
            sim_name=args.sim_name,
            base_dir=args.base_dir,
            rfmix_eu_ref=args.rfmix_eu_ref,
            rfmix_na_ref=args.rfmix_na_ref,
            rfmix_af_ref=args.rfmix_af_ref,
            simple_af_ref=args.simple_af_ref,
            simple_nd_ref=args.simple_nd_ref,
            force=args.force,
        )
        return

    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
