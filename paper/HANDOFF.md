# Appendix E — handoff notes

Draft appendix reporting the full re-run grid: 33 methods, all four LargeST
subsets, both 2018 and 2019 (eight region-year cells, all complete). Written to
be liftable into the main text.

*Last updated 2026-10-09, after CA 2019 was completed.*

---

## 1. Files

The Overleaf mirror is `paper/TMLR_Traffic_benchmark/`; the copies of
`appendix_extended.tex` / `appendix_tables.tex` directly under `paper/` are kept
in sync with it.

| File | What it is | Hand-edited? |
|---|---|---|
| `TMLR_Traffic_benchmark/appendix_extended.tex` | The section: prose, 5 takeaway boxes, 2 figures (7 panels), 1 rank table | yes |
| `TMLR_Traffic_benchmark/appendix_tables.tex` | 8 result tables + coverage table | **no — generated** |
| `results_grid.csv` | The grid, one row per (region, year, method); 237 rows | yes (transcribed) |
| `appendix_refs.bib` | 6 bib entries (already merged into `ref.bib`) | yes |
| `TMLR_Traffic_benchmark/figs/*.pdf` | 7 figure panels (5 hour-of-day, 2 per-sensor) | — |
| `../scripts/analysis/make_appendix_tables.py` | Generates `appendix_tables.tex` from the CSV | — |
| `REPORT_TMLR.md` | Report on the acmart → TMLR template migration | — |

**To update numbers:** edit `results_grid.csv`, then

```bash
python3 scripts/analysis/make_appendix_tables.py \
    --out paper/TMLR_Traffic_benchmark/appendix_tables.tex
```

Do not hand-edit `appendix_tables.tex`. Numbers quoted in the prose are **not**
regenerated and must be re-checked by hand — every time a row was added so far,
at least one prose number moved (ranks, Friedman χ², Spearman ρ).

**Avg column convention.** GNN rows and HL carry the engine's
`Average Test` line (mean over steps 1–12); all other rows use
mean(H3, H6, H12). This is deliberate and matches the source spreadsheet.

---

## 2. What is new relative to the current draft

1. **Two model families the main text does not cover**: deep state-space models
   (Linear SSM, S4, LRU, Mamba-v2) and linear forecasters (DLinear, NLinear).
   The SSMs are the decisive control for the graph comparison — task-trained
   like the GNNs, but per-sensor and with no adjacency.
2. **A second year (2018)**, turning the single-year tables into a replication.
3. **Statistical testing** over the eight region-year cells (Friedman + post-hoc,
   pre-specified paired Wilcoxon tests), and a per-sensor paired test on SD 2018.
4. **Two diagnostic figures** on a wider profiling window.

---

## 3. Findings, ranked by how much weight they can carry

**Strong — safe for the main text.**

- `Sundial` is first in **all eight cells** on Avg MAE, and also at H3 and H12
  individually (Wilcoxon p = 0.0078 vs the best graph model, 8/8, mean margin
  5.05 MAE; same p vs the best SSM, margin 2.92). `Sundial`, `Moirai-2`,
  `Chronos-Bolt` and `Kairos` take the top four places in every cell.
- The best SSM beats the best GNN in **7 of 8 cells** (p = 0.023), and in
  **6 of 6** on the three larger subsets (p = 0.031 — significant, by the
  narrowest margin N = 6 allows).
- The gap between the best GNN and the best SSM **widens monotonically with
  network size**, and both years agree at every step: level on SD (716
  sensors), 8–10% behind on GBA (2,352), 7–24% on GLA (3,834), 19–29% on CA
  (8,600).
- On GBA and GLA the best graph-free model beats every executable GNN at every
  horizon (12/12 comparisons).
- **Accuracy and scalability are in tension inside the graph family**: `DGCRN`,
  the best GNN on SD in both years, cannot run on GLA or CA; `D2STGNN` cannot
  run beyond SD; `STGODE` runs nowhere. Qualifier: `AGCRN` is second on SD and
  does scale — a tension, not an exclusion.
- Horizon stability (H12/H3 MAE of each family's best member): FM 0.97–1.33 <
  SSM 1.22–1.52 < GNN 1.36–1.97 < ensembles/ML 1.5–2.4.
- Year-over-year rank stability: Spearman ρ = 0.98 (SD), 0.97 (GBA), 0.95
  (GLA), 0.87 (CA). No rank movement changes which family leads.

**Moderate — usable with the caveat stated.**

- **The pretrained family is not uniform.** `TimesFM` ranks 9th–14th and loses
  to the best SSM in all 8 cells, to the best GNN in 4, to a linear layer in 4,
  despite being ~4× `Sundial`'s size. `Moirai-Small` is 2.0–2.6× worse than
  `Sundial` everywhere. Neither pretraining nor scale is sufficient on its own.
- **CA does not replicate across years for the classical ensembles.** In CA 2018
  the best GNN (`GWNET` 22.29) loses to the weighted ensemble (21.70); in CA
  2019 the best GNN (`AGCRN` 21.31) beats every ensemble (best 31.62). The swing
  comes from the ensembles' inputs (`ridge` 24.11 → 38.33, `seasonal_naive`
  34.85 → 45.56), not from the graph models. The GNN-vs-SSM deficit *does*
  replicate on CA. This is why CA has the lowest year-over-year ρ.
- On SD 2018, three of six FMs have H12 MAE below H3 MAE, four of six on RMSE.
  The two exceptions are the two weakest FMs (`TimesFM`, `Moirai-Small`), so the
  inversion tracks forecast quality, not pretraining. RMSE falling too rules out
  MAE hedging.

**Weak / negative — reported because it constrains the other claims.**

- The omnibus post-hoc test **still cannot** separate the pretrained and graph
  families at α = 0.05: rank gap 2.75 vs Bonferroni–Dunn CD 2.85 (Nemenyi 3.19).
  The gap narrowed from 0.34 to 0.10 when CA 2019 was added; the limit is
  N = 8 datasets, not effect size.
- Per-sensor control (SD 2018): `LRU` vs `ASTGCN` — `LRU` has the lower mean MAE
  but wins on only 330/716 sensors (46.1%), median runs the other way, Wilcoxon
  p = 0.092. Contrast `Moirai-2`, which beats `ASTGCN` on 716/716 sensors and
  `LRU` on 708/716 (both p < 10⁻¹⁰⁰). A mean gap can be uniform or carried by
  a few sensors; only the pretrained model's is uniform here.

---

## 4. Conflicts with the current main text — these need a decision

1. **"GNNs lead at the shortest horizon, FMs at longer ones."** Does not survive
   the wider grid: `Sundial` is first at H3 as well, in all eight cells. Narrow
   the claim to SD or drop it.
2. **Hour-of-day figure (main text Fig. 4a/4b).** The main text reads the
   morning spike as characteristic of recent-driven FMs. On the appendix's
   common 500-window profile, four of five models peak at 05:00 but `Moirai-2`
   peaks at 07:00, and at 05:00 `Moirai-2` (22.70) is far below `GWNET` (30.62),
   `LRU` (43.97), `ASTGCN` (45.02) and `DLinear` (55.03). The spike is real for
   `Moirai-2` but later and smaller than for any model it is contrasted with.
3. **The main-text tables are a different, earlier run.** Deterministic baselines
   match the re-run (e.g. SD `HL` H3 33.61 in both, `DCRNN` H3 17.14 in both),
   so it is the same pipeline, but other rows differ by up to ~10 Avg MAE
   (`Moirai-Small`, `Linear Regression`, `ASTGCN`). Decide which run the paper
   reports.
4. **Models present in one run but not the other.** The main text has
   `Chronos-2`, `LSTM`, and values for `STGODE` and `FlowState`; in the re-run
   `STGODE` and `FlowState` OOM on every subset. A reviewer will ask why the
   appendix says STGODE cannot run while the main text reports it. If the main
   text's GNN rows were taken from LargeST (Liu et al. 2023) rather than run
   locally, that must be cited.
5. **Tables 1 and 3 in the main text share identical GNN H3/H6/H12 blocks** (SD
   and CA), differing only in Avg. One is presumably a copy-paste error.

---

## 5. Data provenance notes

- **CA 2019 SSM and Linear rows come from the spreadsheet, not from
  `/scratch/.../{model}_CA2019_ctx96_w500/`.** Those scratch directories are a
  *different* training run: they do not match the spreadsheet for CA 2018 either
  (e.g. Linear SSM H3 16.09 vs 15.49). Do not regenerate these rows from that
  path.
- **Rows that could not be cross-checked against a source file** (results
  purged from /scratch): CA 2019 `Mamba-v2`, `Chronos-Bolt`, `Kairos`,
  `Moirai-2`, `Sundial`. They are taken from the spreadsheet as-is.
- **Cross-checked against logs/CSVs, exact match:** every CA 2019 GNN row
  (including `ASTGCN`, run independently by both us and the senior with
  identical results), HL, all ML/simple baselines, all three ensembles,
  `Moirai-Small`, `TimesFM`.
- Four digit-transposition typos in an earlier version of the spreadsheet were
  found and fixed (LightGBM H3 MAPE, Mean H12 MAPE, AGCRN H12 MAPE, GWNET Avg
  RMSE).

---

## 6. Limitations to state if this moves to the main text

- One training run per cell; the tests treat run-to-run variance as zero and
  only ask whether the ordering is consistent across region-years.
- The two figures use a wider profiling window (500 windows, 2018-10-30 to
  12-31) than the tables. All models are scored on identical windows, so the
  comparison between them is valid, but absolute values are not table values.
- FlowState, STGODE, D2STGNN (beyond SD) and DGCRN (beyond SD/GBA 2018) OOM;
  shown as "-".

---

## 7. Open items

- [ ] Verify the 6 bib entries from `appendix_refs.bib` against Google Scholar
      (author lists and venues were written from memory).
- [ ] Decide the five conflicts in §4.
- [ ] Delete the unused `figs/node_error_astgcn_lru.pdf` from Overleaf (superseded;
      nothing references it).
- [ ] Consider dropping the coverage table: every results table already shows
      "-" for unavailable methods, so it only adds the reason.
- [x] ~~Add TimesFM rows.~~ Done.
- [x] ~~Add CA 2019 (all families) and TimesFM on CA.~~ Done 2026-10-09; tables
      regenerated, all prose numbers re-verified (33 checks).
- [x] ~~Title updated to `33 Methods × 4 Regions × 2 Years`.~~ Done.
