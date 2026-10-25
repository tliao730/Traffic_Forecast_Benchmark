# TMLR 模板轉換 — 進度報告

**日期**：2026-09-22
**範圍**：acmart (`sigconf`) → TMLR 模板轉換，外加 appendix 資料更新
**檔案位置**：`paper/TMLR_Traffic_benchmark/`（Overleaf 專案的本機副本）

---

## TL;DR

模板已轉換完成，`main.tex` 可以直接上傳 Overleaf 編譯。
本機沒有 LaTeX 環境，**尚未實際編譯驗證**，第一次 Recompile 可能還有殘留錯誤。

下面 **§1 有四件事需要學長決定**，其餘是已完成事項與備查紀錄。

---

## 1. 需要學長決定或確認的事項

### 1.1 ⚠️ 匿名檢查（最高優先，會被 desk reject）

TMLR 官方模板原文：

> Authors must not appear in the submitted version.
> **Non-anonymous submissions will be rejected without review.**

`tmlr.sty` 在不加 `[accepted]` 選項時會**自動隱藏 `\author{}` 的內容**，印出
「Anonymous authors / Paper under double-blind review」，這部分已經處理好。

但**樣式檔管不到正文內容**，以下需要人工掃過，我無法代為判斷：

- [ ] Acknowledgements 段落（實驗室、經費來源、指名感謝）
- [ ] GitHub / 個人網站 / 專案頁面連結
- [ ] 自我引用的寫法：正文若有「our previous work \citep{...}」需改成第三人稱
- [ ] 提到具體機構或叢集名稱（USC、DENSO、Delta 等）
- [ ] 圖片檔案本身的內嵌路徑或浮水印

**需要學長確認**：哪些 citation 是我們自己的論文，需要改寫成中性敘述。

### 1.2 寬表格在單欄版面下的處理

TMLR 是**單欄**，`\textwidth` = 6.5 吋；acmart `sigconf` 的 `table*` 是跨欄約 7 吋。

Appendix 的 8 張表是 **14 欄 × 33 列**，靠 `\resizebox{\textwidth}` 縮放。
換到單欄後字級會再縮小約 7%，**可能會小到不易閱讀**。

**需要學長決定**（建議先編出來看 PDF 再決定）：
- (a) 維持現狀
- (b) 用 `lscape` 橫放
- (c) 拆成兩張表（H3/H6 一張、H12/Avg 一張）

TMLR 沒有頁數限制，所以 (b)(c) 都不會有篇幅壓力。

### 1.3 Caption 粗體設定

已加入：

```latex
\captionsetup{labelfont=bf}        % 「Figure 1:」粗體，說明文字維持正常
\captionsetup[sub]{labelfont=bf}
```

`tmlr.sty` 本身沒有設定 caption 格式，所以這是**偏離期刊預設**的自訂。
粗體標號是 ML 領域通行作法，風險低，但若學長希望完全照 TMLR 預設，刪掉這兩行即可。

### 1.4 Appendix 與正文的既有衝突（沿用先前紀錄）

`paper/HANDOFF.md` §4 已列出 appendix 結論與現行正文 §4 的五處衝突，
其中最需要處理的是：

> 正文的分工結論（graph models 佔優短 horizon、foundation models 佔優長 horizon）
> 在完整 grid 下**不成立**。`Sundial` 在全部 8 個 region-year cell 的 Avg MAE、
> H3、H12 **三個指標都是第一**。

這件事跟模板轉換無關，但如果要投 TMLR，建議一併處理。

---

## 2. 模板轉換：已完成的改動

### 2.1 Preamble（`main.tex` 第 1–306 行整段重寫）

| 類別 | 改動 |
|---|---|
| Document class | `\documentclass[sigconf,review]{acmart}` → `\documentclass[10pt]{article}` + `\usepackage{tmlr}` |
| 刪除 ACM 專屬指令 | `\setcopyright` `\copyrightyear` `\acmYear` `\acmDOI` `\acmISBN` `\ccsdesc` `\keywords` `\orcid` |
| 刪除 ACM 區塊 | `\begin{CCSXML}...\end{CCSXML}`、8 組 `\affiliation{\institution{...}}` |
| 作者區塊 | 8 個獨立 `\author{}` 合併為 **1 個**，改用 JMLR 語法（`\name` / `\email` / `\addr` / `\AND`） |
| Abstract 位置 | 從 `\maketitle` **前**移到 **後**（acmart 與 article 的順序相反） |
| 套件重複 | `tcolorbox` 載入兩次且選項衝突（`{}` vs `[most]`）→ 只留 `[most]`；`booktabs`、`xcolor` 各刪一次重複 |
| 套件衝突 | 移除 `\usepackage{algorithmic}`（與 `algpseudocode` 互搶 `\State`、`\Comment`） |
| 補上套件 | `hyperref`、`url`、`caption`（acmart 原本自動附送，`article` 不會） |

### 2.2 正文

| 改動 | 數量 | 說明 |
|---|---|---|
| `\cite{}` → `\citep{}` | 53 | TMLR 用 natbib author-year；`\cite{}` 產生不帶括號的「Ansari et al. (2024)」 |
| `\cite{}` → `\citet{}` | 1 | 第 569 行「Following the setup of Liu et al. (2023)」屬主詞位置 |
| 全形波浪號 `～` (U+FF5E) → `~` | 1 | Introduction 內，pdflatex 會直接中斷 |
| `\bibliographystyle{plainnat}` → `{tmlr}` | 1 | |

### 2.3 版面

- 正文 hour-of-day MAE 兩張圖（ASTGCN / Moirai2）原本是兩個獨立 `figure`，
  中間隔了一整個 CA 小節 → **合併為一個 `figure` + 兩個 `subfigure` 並排**。
  兩個原始 `\label` 都保留，第 1085 行的 `\ref` 範圍引用不受影響
  （編號由 Figure 4、5 變成 Figure 4a、4b）。
- 標題斷行點從 `and\\` 移到冒號後（主/副標題分開）。

### 2.4 已驗證項目

- ✅ `tmlr.sty` / `tmlr.bst` / `fancyhdr.sty` / `math_commands.tex` 已在專案根目錄
- ✅ 無殘留未註解的 acmart 專屬指令
- ✅ 無套件重複載入
- ✅ appendix 引用的 7 張圖檔全部存在
- ✅ appendix 需要的 6 筆 bib entry（`gu2022s4`、`gu2022s4d`、`orvieto2023lru`、
      `dao2024mamba2`、`demsar2006`、`zeng2023transformers`）`ref.bib` 內都有
- ✅ 括號與環境配對平衡
- ✅ 原始 acmart 版本備份於 `main_ACM_BACKUP.tex`

### 2.5 尚未驗證

- ❌ **實際編譯**：本機（Delta）沒有 LaTeX，必須在 Overleaf 上驗證
- 第一次 Recompile 需連按 **三次**，交叉引用（`Table ??`）才會解析完成

---

## 3. 不影響編譯的已知問題

### 3.1 `ref.bib` 有 4 組重複 key

| Key | 狀態 |
|---|---|
| `cao2020spectral` | 內容不同（`@article` vs `@inproceedings`，同一篇 NeurIPS 2020） |
| `merrill2024language` | 內容不同 |
| `yang2025foundation` | 內容不同 |
| `wang2025chattime` | 完全相同 |

BibTeX 遇到重複只會**警告**並採用**第一筆**，不會中斷編譯。
以 `cao2020spectral` 為例，第一筆是 `@article` 版（含 volume/pages），是正確的引用形式。

### 3.2 檔名拼字

`images/time_interval/Moiria2_hourly_mean_abs_error.pdf` — `Moiria` 應為 `Moirai`。
原始程式碼註解中已有人標註。未更動，因為改名需同步改圖檔與引用，風險大於效益。

---

## 4. Appendix 資料更新（與模板轉換獨立）

### 4.1 CA 2018 補齊

`TimesFM` 與 `Moirai-Small` 的 CA 結果已完成並填入：

| Method | H3 MAE | H6 MAE | H12 MAE | **Avg MAE** |
|---|---|---|---|---|
| TimesFM | 15.23 | 18.15 | 22.46 | **18.61** |
| Moirai-Small | 21.70 | 30.42 | 42.49 | **31.54** |

（Avg 沿用既有慣例 mean(H3, H6, H12)）

CA 2018 現為 **29 列**，Table 9 覆蓋表中兩個模型已全部標記為 ✓。

> **後續更新（2026-10-09）**：CA 2019 已全部完成並加入（29 列），grid 現為 8 個 cell。
> 統計檢定、排名與所有內文數字已依 8 格重算並逐項驗證，詳見 `HANDOFF.md`。

### 4.2 因此連帶修正的敘述

| 原文 | 改為 | 原因 |
|---|---|---|
| TimesFM「在它跑過的**六格**全敗給最佳 SSM…被單層線性擊敗**兩次**」 | 「**七格**…**三次**」 | CA 2018 的 TimesFM 18.61 輸給 DLinear 17.79 |
| Friedman「全部 **27** 個模型，χ²=167.3，p<10⁻²¹」 | 「**29** 個，χ²=**181.7**，p<10⁻²³」 | 完整跑完七格的模型數增加 |
| SD 2018 橫向反轉「**五個** FM 中三個…唯一例外是 TimesFM」 | 「**六個** FM 中三個…兩個例外是最弱的兩個：TimesFM (1.62) 與 Moirai-Small (1.98)」 | **既有錯誤**：原文漏算 Moirai-Small |

最後一項是複驗時發現的原有疏漏，修正後論證反而更強——兩個例外正好是最弱的兩個 FM，
直接支持「反轉現象跟預測品質相關、與是否預訓練無關」的結論。

### 4.3 複驗仍成立（未更動）

- TimesFM 排名區間 9–14 名
- Moirai-Small 相對 Sundial 為 2.0–2.6 倍（含 CA 後為 2.03–2.62）
- 家族代表的 rank 表、k=7 的 Friedman 與 Bonferroni–Dunn 數值完全不變
- CA 2018 的 GWNET 22.29 / LRU 17.24 / DLinear 17.79 / weighted 21.70 / Sundial 14.29

### 4.4 Appendix 章節標題

最終標題為 **`Extended Results: 33 Methods × 4 Regions × 2 Years`**。

（CA 2019 完成前曾暫用 `33 Methods × 7 Region-Years`，避免與當時 Table 9 的 7 欄、
§E.4 的 N=7 矛盾；CA 2019 補齊後已改回 4×2。）

---

## 5. 待辦

| 項目 | 負責 | 備註 |
|---|---|---|
| Overleaf 實際編譯驗證 | 我 | 上傳後執行，有錯再修 |
| 匿名檢查（§1.1） | **需學長協助** | 自我引用需學長判斷 |
| 寬表格處理方式（§1.2） | **需學長決定** | 建議先看 PDF |
| ~~CA 2019 跑完後補表~~ | 我 | ✅ 已完成：N 7→8，Wilcoxon p 0.016→0.0078；三大 subset 6/6 p=0.031 達顯著 |
| Appendix 與正文衝突（§1.4） | **需學長決定** | 詳見 `HANDOFF.md` §4 |
| `ref.bib` 去重 | 低優先 | 僅警告，不影響編譯 |

---

## 附錄：檔案清單

| 檔案 | 說明 |
|---|---|
| `TMLR_Traffic_benchmark/main.tex` | 已轉換，可上傳 |
| `TMLR_Traffic_benchmark/main_ACM_BACKUP.tex` | acmart 原版備份 |
| `TMLR_Traffic_benchmark/appendix_extended.tex` | Appendix 正文（約 3,490 字） |
| `TMLR_Traffic_benchmark/appendix_tables.tex` | 由腳本生成，**請勿手動編輯** |
| `paper/results_grid.csv` | 表格資料來源（206+ 列） |
| `scripts/analysis/make_appendix_tables.py` | 表格生成腳本 |
| `paper/TMLR_MIGRATION.md` | 轉換步驟說明（操作手冊） |
| `paper/HANDOFF.md` | Appendix 交接說明（findings、衝突、限制） |
