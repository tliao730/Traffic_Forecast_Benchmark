# acmart → TMLR 轉換步驟

目標：把 Overleaf 上的論文從 ACM `sigconf` 格式換成 TMLR 格式。

官方模板來源：<https://github.com/JmlrOrg/tmlr-style-file>
作者指南：<https://www.jmlr.org/tmlr/author-guide.html>

---

## Step 0 — 先備份（最重要，別跳過）

在 Overleaf 專案頁面：**Menu → Copy Project**，命名成 `...-acm-backup`。

換模板會動到整個 preamble，出錯時要有東西可以回去對照。Overleaf 的 History
雖然能還原，但有一份完整可編譯的舊版並排著看，除錯會快很多。

接下來所有操作都在**原專案**上做，備份那份不要碰。

---

## Step 1 — 上傳 TMLR 樣式檔

把 `paper/tmlr_style/` 裡這四個檔案上傳到 Overleaf 專案的**根目錄**
（跟 `main.tex` 同一層，不要放進子資料夾）：

```
tmlr.sty            ← 樣式主檔
tmlr.bst            ← 參考文獻格式
fancyhdr.sty        ← tmlr.sty 需要
math_commands.tex   ← 選用的數學巨集
```

`OFFICIAL_EXAMPLE.tex` 不用上傳，那是官方範例，留著本機對照用。

---

## Step 2 — 換掉 preamble

### 2a. 刪掉所有 ACM 專屬指令

這些在 `article` class 下都不存在，留著一定報錯：

```latex
\documentclass[sigconf,review]{acmart}   ← 整行換掉
\acmConference{...}                      ← 刪
\acmBooktitle{...}                       ← 刪
\acmDOI{...} \acmISBN{...}               ← 刪
\copyrightyear{...} \acmYear{...}        ← 刪
\setcopyright{...}                       ← 刪
\settopmatter{...}                       ← 刪
\begin{CCSXML} ... \end{CCSXML}          ← 整塊刪
\ccsdesc[...]{...}                       ← 刪
\keywords{...}                           ← 刪（TMLR 不用）
\received{...}                           ← 刪
```

### 2b. 新的開頭

```latex
\documentclass[10pt]{article}
\usepackage{tmlr}
% 被接受後改用： \usepackage[accepted]{tmlr}
% 貼 arXiv 時改用： \usepackage[preprint]{tmlr}
```

### 2c. 補上 acmart 以前「免費附送」的套件

**這步最容易漏。** acmart 自動載入了很多套件，`article` 不會。
現有的表格用了 `\toprule` / `\midrule` / `\bottomrule`，沒有 `booktabs` 會直接爆：

```latex
\usepackage{graphicx}
\usepackage{booktabs}     % ← \toprule \midrule \bottomrule（一定要）
\usepackage{multirow}     % ← \multirow
\usepackage{pifont}       % ← \ding{51} \ding{55}
\usepackage{subcaption}   % ← subfigure
\usepackage{amsmath,amssymb}
\usepackage{tcolorbox}    % ← insightbox
\usepackage{hyperref}
\usepackage{url}
```

其中 `tcolorbox` 只能載入**一次**，而且要帶齊 `insightbox` 需要的選項。
（舊版 preamble 有重複載入 tcolorbox 的問題，順便一起修掉。）

`insightbox` 的自訂定義原封不動搬過來即可，它不依賴 acmart。

---

## Step 3 — 改標題和作者區塊

acmart 和 TMLR 的作者語法完全不同，不能直接沿用。

TMLR 用 JMLR 語法：

```latex
\title{Traffic Forecasting in Practice: ...}

\author{\name First Author \email first@example.edu \\
      \addr Department, University
      \AND
      \name Second Author \email second@example.edu \\
      \addr Department, University}

\begin{document}
\maketitle

\begin{abstract}
...
\end{abstract}
```

**注意順序**：acmart 是 `abstract` 寫在 `\maketitle` **前面**，
TMLR / article 是寫在**後面**。順序錯了 abstract 不會出現。

### ⚠️ 匿名（會被 desk reject 的點）

官方模板原文：

> Authors must not appear in the submitted version.
> **Non-anonymous submissions will be rejected without review.**

用 `\usepackage{tmlr}`（不加 `accepted` / `preprint`）就會自動隱藏
`\author{}` 的內容，所以作者區塊照上面寫沒問題。

但這些**要自己檢查**，樣式檔管不到：

- [ ] Acknowledgements 段落（提到實驗室、指名感謝的人）→ 投稿版先刪掉
- [ ] GitHub / 個人網站連結
- [ ] 正文寫「our previous work \citep{我們自己的論文}」→ 改成第三人稱
- [ ] 提到具體機構、叢集名稱（例如 Delta、USC）
- [ ] PDF 檔案內容裡的路徑

---

## Step 4 — 改參考文獻

```latex
\bibliographystyle{ACM-Reference-Format}   ← 舊
\bibliographystyle{tmlr}                   ← 新

\bibliography{ref}   ← 檔名不用改，還是你的 ref.bib
```

`ref.bib` 本身**完全不用動**。

### 引用指令

`tmlr.sty` 載入 `natbib`，所以：

| 用法 | 產出 | 什麼時候用 |
|---|---|---|
| `\citep{demsar2006}` | (Demšar, 2006) | 句子裡括號引用（**大多數情況**） |
| `\citet{demsar2006}` | Demšar (2006) | 當作句子主詞，例如「\citet{x} 指出…」 |

ACM 是數字格式 `[12]`，`\cite{}` 到處用都沒差；
TMLR 是 author-year，用錯會讀起來很怪。

**appendix 那邊我已經改好了**（5 處 `\cite` → `\citep`）。
正文的 `\cite{}` 要請學長一起掃一遍。

---

## Step 5 — Recompile

按 **Recompile**。第一次一定會有一堆錯，正常。

依序處理：

1. **先看第一個 error，不要看 warning。** LaTeX 的錯誤會連鎖，
   第一個修好之後後面常常自己消失一半。
2. `Undefined control sequence` → 多半是漏載套件（回 Step 2c）或
   ACM 專屬指令沒刪乾淨（回 Step 2a）。
3. 全部 error 清乾淨後，**再按兩次 Recompile**。
   表格和圖的交叉引用（`Table ??`）需要兩輪才會解析。

---

## Step 6 — 版面檢查

TMLR 是**單欄**，acmart sigconf 是雙欄。內容不會壞，但要看一下：

- **寬表格**：`\resizebox{\textwidth}` 會自動縮到新的 `\textwidth`（6.5in，
  比原本雙欄跨欄的略窄），字會再小一點。如果 33 列 × 14 欄縮到看不清楚，
  可以考慮橫放（`lscape`）或拆表。編出來看過再決定。
- **`table*` / `figure*`**：在單欄文件下會自動退回成一般 `table` / `figure`，
  **不會報錯也不會壞版**，所以第一次轉換時可以先不動。
  等整份編得出來之後再改成不帶星號的版本（純粹是寫法乾淨）。
- TMLR **沒有頁數限制**，所以 appendix 很長不是問題。

---

## 給我檔案的方式

我打不開 Overleaf（要登入的網頁應用），給我連結沒用。我需要檔案本身。

最方便的做法：

1. Overleaf → **Menu → Download → Source (.zip)**
2. 在 VSCode 左邊的檔案總管，把 zip **直接拖進** `TrafficFM/paper/` 資料夾
   （VSCode Remote 支援拖曳上傳到遠端）
3. 跟我說一聲，我解開來直接幫你改 `main.tex`，改好你再傳回 Overleaf

只要 preamble 的話，直接貼在對話裡也可以，但整包給我的話我能一次把
Step 2～4 全部做完並交叉檢查，你只要負責上傳和按 Recompile。
