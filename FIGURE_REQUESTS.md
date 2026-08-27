# Figure requests for `I2I-Stain-Zoo`

Two supplement figures the WACV paper wants, written to be executed against
`/Users/qasim/Desktop/Hoehme_Git/Qasim/I2I-Stain-Zoo/`. Companion to
`analysis_requests.md`, which specified the W-2, W-29 and W-30 *analyses*; those have all
run and their numbers are in the paper. **This file is about plots, not numbers.**

**Neither figure changes a result.** F-1 re-renders a plot that already exists. F-2 is new
but illustrative, showing on real tissue what the text already asserts.

**Both are supplement-bound.** The main body is full at 8 pages, so neither may cost main
text beyond one pointer sentence. **Point at the appendix *section*, never at a figure
number** — a main-to-supplement figure reference resolves to the wrong float silently, and
`8_discussion.tex:15` records the project hitting this.

---

## 0. What was verified in the repo before writing this

| Fact | Location |
|---|---|
| The W-29 plot already exists and is produced by a committed script | `stability_data_exposure.py:438-502` |
| It writes PNG at `dpi=200`, `figsize=(9.0, 3.5)` | `stability_data_exposure.py:443, 502` |
| Panel titles are working notes, and the suptitle prints the raw descriptor key | `:480, :495, :499` |
| A paper style helper exists, one WACV column is `3.281 in` | `make_paper_figures.py:91, 136-142` |
| Paper figures are written as **PDF**, `bbox_inches="tight"`, `pad_inches=0.01` | `make_paper_figures.py:222` |
| Region geometry lives in `uncertainty_phi/regions.py` | referenced in `CLAUDE.md` |
| A per-region map plotter already exists | `plot_uncertainty_heatmap.py` |

**Assumed, please correct if wrong:** `make_paper_figures.py`'s `style()` and `COLUMN_IN`
can be imported or copied by another script, and the existing figure assets in
`Manuscript/.../figures/` were produced under that convention. Their native widths are
211–234 pt, consistent with one column.

---

## 1. Conventions both figures must follow

These are not style preferences. Each one is a decision the manuscript already made, and a
figure that breaks one contradicts the text it sits beside.

### Use the paper's symbols, not paraphrases

`PAPER_FIGURES.md` §2a fixes the register and it still holds. The traps that catch every
new plot:

| Write | Never write |
|---|---|
| $\sigma$ | "uncertainty (SD)", "ensemble std" |
| $e$ | **"MAE"**, "MAE-CPA", "\|error\|" |
| $\mu$ | "mean prediction", "predicted CPA" |
| $\rho$, to three decimals | "corr", "r" |
| CPA | `task_specific_value`, "descriptor", "phi" |
| cycle-reconstruction **residual** | cycle-reconstruction "error" |

### Terminology settled after `PAPER_FIGURES.md` was written

Four renames landed in the manuscript on 2026-08-26 and 2026-08-27. **`PAPER_FIGURES.md`
predates all four**, so following it alone is no longer sufficient.

| Now | Was, and must not reappear |
|---|---|
| **pixel-based protocol** | "conventional protocol" |
| **task-based protocol** | "proposed protocol" |
| **tissue mask** | "tissue footprint" |
| **liver arm**, **kidney arm**, **both organ arms** | a bare "arm". The word is reserved for organ cohorts, so real-versus-generated is "the real and the generated images" |

Also: the resource is named **HistoCal**, and the released cases are **cases**, with
whole-slide-image counts in parentheses if both are given. Never switch to slide counts as
the unit.

### No title inside the figure

LaTeX captions the float. A figure carrying its own title double-prints it, which is why
F-1 loses its suptitle. Panel labels are fine where a multi-panel figure needs them, as
plain noun phrases.

### Caption structure, if you draft one

The manuscript's captions run **what it is, then what you see in it, then how to read it**,
and state the shape of the curves before the construction. The author's standing reason is
that some readers come back to a paper and look only at the plots and their captions. Ship
a draft caption with each figure and the manuscript will adapt it, since the caption is
also where a value with no float of its own gets anchored.

### Asset conventions

| Property | Value | Source |
|---|---|---|
| Format | PDF, `bbox_inches="tight"`, `pad_inches=0.01` | `make_paper_figures.py:222` |
| One-column width | `COLUMN_IN = 3.281` in | `make_paper_figures.py:91` |
| Style | `style(7.0)` | `make_paper_figures.py:136` |
| Native width of existing assets | 211 to 234 pt | measured from `figures/*.pdf` |

**Do not render wider than the column and rely on LaTeX to scale it down.** The two
reliability plots are natively 213.8 pt against a 236.5 pt column and are now included at
`0.85\columnwidth` so they render near their design size. A figure authored at 9 in and
scaled into a column loses roughly two thirds of its label size.

---

## F-1 · Re-render the data-exposure stability figure for publication

### The question

§8's Limitations discloses that the data-exposure component is the least determined of the
three, its median share moving between `0.282` and `0.562` under leave-one-subset-out while
being stable across cases and seeds. **Right now that disclosure is words only.** The
existing plot shows it directly, the fold-4 replicate sitting far below four folds clustered
on `0.508`, and a reviewer who meets that unprompted is in a better position than one who
finds it. This was SH-7's question.

**The science is done and must not change.** This is a rendering job.

### Inputs

Unchanged from the run already performed:

```
per_region : /work2/bz66izin-UC_project/ID_HE/phi_uncertainty/agg_phi/per_region.csv
descriptor : task_specific_value
folds      : fold1 … fold5, 10 seeds each, n_regions = 2850
```

### What to change

| # | Change | Reason |
|---|---|---|
| 1 | **PDF, not PNG.** `format="pdf"`, `bbox_inches="tight"`, `pad_inches=0.01` | Every other paper figure is vector. PNG at `dpi=200` will look soft beside them |
| 2 | **Label the quantity CPA**, not `task_specific_value` | `task_specific_value` is an internal key. The paper says CPA and a reader will not connect the two |
| 3 | **Drop the suptitle entirely** | The caption carries the title in LaTeX. A figure with its own title double-prints |
| 4 | **Replace both panel titles** with plain noun phrases, or drop them and let the axes speak. Current text is `"how far the headline moves under each cut"` and `"a property of the design, not the estimator"` | Working notes, not paper register. §8's headers were rewritten for exactly this reason (W-17) |
| 5 | **Size for one column**, `width = COLUMN_IN`, and apply `style(7.0)` | At `figsize=(9.0, 3.5)` scaled into a 3.28 in column every label drops to roughly a third of its designed size |
| 6 | Keep both panels if they fit legibly at column width. **If they do not, ship the left panel only** | The left panel carries the finding. The right panel's df point is already stated in words in §8 and can stay there |

### Keep exactly as they are

- The fold-4 replicate at `0.282` visible and unhighlighted. **Do not clip the y-axis to
  hide it, do not mark it as an outlier.** Its being visible is the entire point.
- The case-bootstrap band and the full-grid line at `0.508`.
- Both seed-subsample groups, `S = 10 → 8` and `10 → 5`.

### Outputs wanted

```
figures/stability_data_exposure.pdf
```

### Acceptance checks

Reproduce, from the re-render, the values already in the paper:

| Quantity | Must be |
|---|---|
| full-grid median share | `0.508` |
| leave-one-subset-out range | `0.282` to `0.562` |
| number of LOSO replicates | 5 |
| seed-subsample spread | below `0.005` |

If any differs, **stop and say so** rather than shipping the figure. The paper quotes these
numbers and they are already committed.

### Likely home

`sec/supp_data.tex`, Additional results, in **What the ensemble's spread is made of**,
beside `fig:dataexposure`. **No new main-text pointer is needed**, since §8's Limitations
already cites `Appendix~\ref{sec:supp-results}` for the leave-one-subset-out range. The
range currently sits in `fig:dataexposure`'s caption and would move to this figure's.

---

## F-2 · The spatial region-mapping figure

### The question

AF asked twice, on his page 6 and page 8, for the spatial relationship to be shown rather
than described. The paper asserts that the region grid is defined once in the H\&E frame and
that both stains are read in it, and that correspondence holds at region scale but not below
it. **Nothing in either document shows this on tissue.** It is the one figure that would let
a reader judge the premise instead of taking it.

It also stands in for a number the paper cannot supply. W-10, the separation between the two
levels, **cannot be obtained** and is disclosed as unavailable, so a picture of what
region-level correspondence looks like is the nearest available evidence.

### What it should show

One case, both stains, with the analysis grid drawn over both.

1. **H\&E whole-slide thumbnail** with the `2048 × 2048` px region grid overlaid, regions
   below the 25% tissue-coverage rule shown as dropped.
2. **The registered Sirius Red** for the same case, same grid, same coordinates.
3. **One region magnified from both stains, side by side.** This is the panel that does the
   work. It should be visibly the same tissue area and visibly *not* the same structures,
   which is precisely the claim §5 makes.

**Choose an ordinary case, not the best-registered one.** If a reader later downloads the
release and finds the figure unrepresentative, that is worse than a plain figure. Say in the
caption which case it is.

### Practical notes

- `plot_uncertainty_heatmap.py` already renders per-region maps over a slide and is probably
  the closest starting point.
- Region geometry, `region_px = 2048`, `SOURCE_MPP = 0.221`, `min_tissue_fraction = 0.25`,
  `drop_partial`, is in `uncertainty_phi/regions.py`.
- The supplement has **no page limit**, so this may be full text width and taller than one
  column. Legibility beats compactness.
- **Anonymity.** The crops must carry no slide label, no barcode, no scanner overlay and no
  filename burned into the image. The review copy is double-blind and whole-slide formats
  embed a label image that routinely photographs handwritten case identifiers.

### Outputs wanted

```
figures/region_mapping.pdf
```

### Acceptance checks

| Check | Why |
|---|---|
| The grid is identical in both stains, same origin and same cell size | If it is not, the figure contradicts §6 |
| Dropped regions are the ones below 25% coverage | Same rule the results use |
| The magnified pair is the same region index in both | The figure's whole claim |
| No identifying text in any panel | Double-blind |

### Likely home

`sec/supp_data.tex`, Cohort details, in **Registration and what it can recover**, which is
where region-level correspondence is described and where the eight-fold fidelity range is
disclosed.

**This one does need a main-text pointer**, one sentence in §5's *Non-adjacent sections* or
§6's *Registration and coordinate frame*, naming the appendix **section**. The manuscript
will write it. Nothing in the paper currently references this figure, so until the pointer
lands the paper is complete without it.

---

## Not requested, and why

**The workflow / processing-chain diagram is cancelled** (author, 2026-08-27). Do not build
it. `workflow_diagram_draft.md` in the manuscript folder is Mermaid, never reached the LaTeX
build, and §6's pointer to it has been removed.

**The W-30 rank-rules plot is deliberately not wanted.** `rank_rules_total.png` exists and is
correct, but in its within-slide panel every rule bunches into one band, so the 1.7-point
advantage of the rank sum over the prediction alone cannot be read off it, and it overlaps
`risk_coverage.pdf`. A reader would see near-identical curves and doubt a claim the text
states carefully with its interval. **A figure that undercuts its own caption is worse than no
figure.** The result stays in the text and in `fig:riskcoverage`'s caption.

**The W-2 shape-factor plots are not wanted either.** The author decided to state the Gaussian
assumption and keep the published `0.80σ` and `0.46σ` lines, and **the measured $\hat\kappa$ is
not reported**. A figure showing it would report it.

---

## Suggested order

**F-1 first.** It is a re-render of a committed script against an input that has not moved,
so it is short, and it closes a disclosure that is currently words alone.

**F-2 second**, and it is the one that needs judgement rather than code. If it turns out
expensive, say so: the paper is complete and consistent without it, nothing references it,
and it would be added along with its own pointer sentence.
