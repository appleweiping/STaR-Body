# CVPR 2027 submission (LaTeX)

Anonymous review version of *STaR-Body: Identity-Preserving Evidence Routing for
Spatiotemporal Human Mesh Refinement*, built on the official CVPR 2027 author kit
(`cvpr.sty` and `ieeenat_fullname.bst` are copied unchanged).

| File | Content |
|---|---|
| `main.tex`, `preamble.tex`, `sec/*.tex` | main paper (8 pages + LLM Usage Statement + references) |
| `supp.tex` | supplementary material (single column, sections S1–S7) |
| `main.bib` | references (30 audited entries from `paper/source/venue_reference_audit.json` + 10 added for C11) |
| `generated/results.tex` | **all new numbers**; written by the analysis script, never by hand |
| `figures/*.jpg` | overview and routing schematics (downscaled from `paper/source/figures/*.png`) |
| `figures/generated/*.pdf` | data figures from `scripts/make_c11_figures.py` (framed placeholder until they exist) |
| `.gitignore` | ignores LaTeX build products and re-includes `figures/*.jpg` (the repository root ignores `*.jpg`) |

## Build

```bash
bash build.sh          # latexmk main.tex, then supp.tex (supp reads main.aux for cross-references)
bash build.sh clean    # remove build products
```

Both documents compile with `latexmk -pdf -interaction=nonstopmode -halt-on-error <file>.tex`;
build `main.tex` before `supp.tex`. `build.sh` also prints the page counts and the page on which
the LLM Usage Statement / references start (the main text must end by page 8).

## How results reach the paper

1. Train, evaluate, lock and test as in `docs/C11_RUNBOOK_ZH.md`.
2. `python scripts/run_c11_matrix.py --matrix configs/c11/matrix.local.json analyze`
   runs `scripts/analyze_results_c11.py`, which writes `evidence/c11/summary.{json,md}` and
   overwrites `paper/latex/generated/results.tex` with macro definitions.
3. `python scripts/make_c11_figures.py --matrix configs/c11/matrix.local.json` writes
   `figures/generated/{ablation,sequences,confidence}.pdf`.
4. `bash build.sh`. Every `\res{key}` without a definition prints a red **TBD**.

Macro keys (see the header of `preamble.tex`):

```
\res{<variant>@<split>@<metric>}         mean over seeds
\res{<variant>@<split>@<metric>@pm}      mean ± std
\res{<variant>@<split>@nseeds}
\res{cmp@<variant>@<control>@<split>@<metric>@diff|ci|wins|p}
split  ∈ {validation, test}
metric ∈ {mpjpe, mpjpehip, pampjpe, pve, pvepelvis, pvehip, accel, accelms}
```

Reference checkpoints appear as `P-parent`, `R-c10-archived`, `R-c00-archived` (seed 0).
Controlled tables use `mpjpe` / `pvepelvis` / `accel`; Table 1 shows our rows under both root
conventions (`mpjpe`/`pvepelvis` and `mpjpehip`/`pvehip`) and Accel in m/s² (`accelms`).
The only literal numbers in the text are archived C10/C00 records (labelled archived, single
seed, validation), the C10-vs-parent bootstrap in Supp. S5 (computed from
`evidence/c10_per_sequence_differences.json`), protocol counts, parameter counts, and
author-reported literature values from `paper/source/published_benchmark_rows.json` and
`paper/source/reported_baselines/literature_reported_3dpw_test.json`.

Per-sequence keys used in Supp. S7 (`cmp@<variant>@<control>@<split>@<metric>@seq@<3DPW sequence
name>`) are emitted by the analysis script for every comparison listed in the matrix.

## Open `\TODO` decisions (red in the PDF)

Results-dependent text (decision rule fixed in advance: a difference is reliable only if its 95%
interval excludes 0 and all seeds agree in sign; see `docs/C11_CVPR_PLAN_ZH.md` §5):

1. Abstract: one-sentence main result (B4 vs B7 / B1 on test).
2. Introduction: closing sentence of the contributions paragraph.
3. Sec. 4.2: where B4 lies relative to WHAM/GVHMR (hip root) and CameraHMR/ReFit/PromptHMR
   (pelvis root); no state-of-the-art claim. Confirm the acceleration limitation with `accelms`.
4. Sec. 4.3: self-referential evidence (A2 vs A0, A2 vs A1); external evidence (B4 vs A0, B4 vs A2);
   identity and routing (B4 vs B1, B4 vs B7) using the three-way decision tree.
5. Sec. 4.4: identity/bias ablations (B2, B3, B5); evidence quality and fusion (C4, C1, B6).
   Drop the C rows and paragraph if the optional C runs were not made.
6. Sec. 4.5: per-sequence and confidence patterns; align Figure 3 caption with the figure script.
7. Conclusion: the answer in one or two sentences.

Facts to confirm:

8. Sec. 3.2 and Supp. S5: run `scripts/inspect_checkpoint_v43.py` on the archived parent, C00 and
   C10 checkpoints and insert the verdict (expected: observation head untrained). Remove the
   `\TODO{confirm}` in the introduction afterwards.
9. Supp. S5: whether re-evaluating the archived checkpoints with the v43 code reproduces the
   archived validation numbers, and which weight field each checkpoint stores.
10. Supp. S2: GPU type, time per run and peak memory.
11. If any tier ran with fewer than three seeds, update "three seeds" in the text and captions
    (the `@nseeds` macros in Supp. S6 show the actual counts).

Figures:

12. Figures 1 and 2 are the C10 schematics: redraw the evidence branch for external keypoints
    (u, c; features u−p, p, −2 log c, c, validity). Figure 1 shows the step scales 2°/5°
    and cumulative 6°/15°; the tight cumulative bounds are 5.2°/13.0°.

Before submission: resolve every `\TODO`, then uncomment `\renewcommand{\TODO}[1]{}` in
`preamble.tex` only as a safety net, and check that no red TBD remains.
