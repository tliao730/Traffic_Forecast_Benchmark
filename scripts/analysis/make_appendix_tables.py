#!/usr/bin/env python3
"""Emit the appendix result tables (one per region/year) in the paper's LaTeX style.

Reads paper/results_grid.csv -- the transcribed run grid, one row per
(region, year, method), holding the nine H3/H6/H12 x MAE/RMSE/MAPE numbers plus
the three Avg columns. Every value is taken from the source sheet verbatim; this
script formats and ranks, it does not compute metrics.

Usage:
    python scripts/analysis/make_appendix_tables.py --out paper/appendix_tables.tex
"""
import argparse, csv, os, sys
from collections import defaultdict

GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper", "results_grid.csv")

REGIONS = [("sd", "San Diego"), ("gba", "Bay Area"),
           ("gla", "Los Angeles"), ("ca", "CA")]
YEARS = ["2019", "2018"]

# Row order in every table. Methods absent from a cell are simply skipped.
FAMILIES = [
    ("Simple Baselines", ["HL", "Naive", "Seasonal Naive"]),
    ("ML Baselines", ["Ridge", "Random Forest", "Linear Regression", "LightGBM"]),
    ("Ensembles", ["Median", "Mean", "Weighted"]),
    ("SSM", ["Linear SSM", "LRU", "S4", "Mamba-v2"]),
    ("Linear", ["DLinear", "NLinear"]),
    ("Foundation Models", ["FlowState", "Chronos-Bolt", "Kairos", "Moirai-2",
                           "Moirai-Small", "TimesFM", "Sundial"]),
    ("GNN-based", ["AGCRN", "ASTGCN", "D2STGNN", "DCRNN", "DGCRN", "DSTAGNN",
                   "GWNET", "STGCN", "STGODE", "STTN"]),
]
ROW_ORDER = [(fam, m) for fam, ms in FAMILIES for m in ms]

COLS = [f"H{h} {m}" for h in (3, 6, 12) for m in ("MAE", "RMSE", "MAPE")] + \
       [f"Avg {m}" for m in ("MAE", "RMSE", "MAPE")]
KEY = {"MAE": "mae", "RMSE": "rmse", "MAPE": "mape"}


def load(path):
    cells = defaultdict(dict)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            vals = {}
            for h in (3, 6, 12):
                for m, k in KEY.items():
                    v = r.get(f"h{h}_{k}", "").strip()
                    vals[f"H{h} {m}"] = float(v) if v else None
            for m, k in KEY.items():
                v = r.get(f"avg_{k}", "").strip()
                vals[f"Avg {m}"] = float(v) if v else None
            cells[(r["region"], r["year"])][r["method"]] = vals
    return cells


def fmt(v, col):
    if v is None:
        return "-"
    return f"{v:.2f}\\%" if "MAPE" in col else f"{v:.2f}"


def render(region, region_name, year, data):
    if not data:
        return None
    rank = {}
    for c in COLS:
        vals = sorted((d[c], m) for m, d in data.items() if d.get(c) is not None)
        rank[c] = [m for _, m in vals[:2]]

    # Every method appears in every table; one that produced no usable result on
    # this subset is shown as "-" rather than dropped, so the composition of each
    # family stays visible from the table alone.
    present = list(ROW_ORDER)
    counts = defaultdict(int)
    for fam, _ in present:
        counts[fam] += 1

    L = [r"\begin{table*}[t]", r"\centering",
         f"\\caption{{Performance Comparison of Forecasting Methods "
         f"({region_name} Data, {year})}}",
         f"\\label{{tab:app_{region}_{year}}}",
         r"\resizebox{\textwidth}{!}{",
         r"\begin{tabular}{c|c|" + "c|" * 11 + "c}",
         r"\toprule",
         "Category & Method & " + " & ".join(COLS) + r" \\"]

    seen = set()
    for fam, m in present:
        if fam not in seen:
            L.append(r"\midrule")
            head = f"\\multirow{{{counts[fam]}}}{{*}}{{{fam}}}"
            seen.add(fam)
        else:
            head = ""
        cells = []
        for c in COLS:
            s = fmt(data.get(m, {}).get(c), c)
            if s != "-" and m in rank[c]:
                s = f"\\textbf{{{s}}}" if rank[c][0] == m else f"\\underline{{{s}}}"
            cells.append(s)
        L.append(f"{head} & {m} & " + " & ".join(cells) + r" \\")

    L += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}"]
    return "\n".join(L)


def coverage(cells):
    live = [(r, y) for r, _ in REGIONS for y in YEARS if cells.get((r, y))]
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Run coverage. \ding{51}~= evaluated; \ding{55}~= no usable result "
         r"for that subset, shown as ``-'' in the tables that follow.}",
         r"\label{tab:app_coverage}",
         r"\resizebox{\textwidth}{!}{",
         r"\begin{tabular}{c|c|" + "c|" * (len(live) - 1) + "c}",
         r"\toprule",
         "Category & Method & " + " & ".join(f"{r.upper()} {y}" for r, y in live) + r" \\"]
    counts = defaultdict(int)
    for fam, m in ROW_ORDER:
        counts[fam] += 1
    seen = set()
    for fam, m in ROW_ORDER:
        if fam not in seen:
            L.append(r"\midrule")
            head = f"\\multirow{{{counts[fam]}}}{{*}}{{{fam}}}"
            seen.add(fam)
        else:
            head = ""
        L.append(f"{head} & {m} & " +
                 " & ".join(r"\ding{51}" if m in cells[c] else r"\ding{55}" for c in live) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=os.path.normpath(GRID))
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    cells = load(args.grid)
    parts = [coverage(cells), ""]
    for region, name in REGIONS:
        for year in YEARS:
            t = render(region, name, year, cells.get((region, year), {}))
            if t:
                parts += [t, ""]

    out = "\n".join(parts)
    if args.out == "-":
        print(out)
    else:
        with open(args.out, "w") as fh:
            fh.write(out)

    for region, _ in REGIONS:
        for year in YEARS:
            d = cells.get((region, year))
            if d:
                print(f"  {region}/{year}: {len(d)} methods", file=sys.stderr)


if __name__ == "__main__":
    main()
