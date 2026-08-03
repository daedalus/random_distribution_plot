# theoretical-pdf-overlay: Add theoretical PDF curves to all histogram panels

**Date:** 2026-08-03
**Context:** random_dist.py — 7×4 grid of distribution histograms

## Problem
The user asked to "add the softmax line to every graph." After clarifying, they meant overlaying the theoretical probability density function (PDF) curve on every histogram panel. The script already had theoretical PDF lines on 5 panels (U1+U2 triangular, order-stat Beta, spacings Beta, extreme-value Exp, CLT Normal) but was missing them on the remaining 19 panels.

## Rejected
- **KDE (kernel density estimate) line** — plausible as a "smooth line" over histograms, but user confirmed they want the theoretical parametric PDF, not a data-driven estimate.
- **Actual softmax function** — the term "softmax" was initially ambiguous; the user confirmed they want the theoretical distribution PDF, not the softmax activation function.

## Approach
Added `from scipy import stats` import, then added theoretical PDF/PMF lines using `scipy.stats` distribution functions to each of the 19 histogram panels that lacked them. For continuous distributions, used `axs[i].plot(x, stats.<dist>.pdf(x, ...), "k--", lw=1.5)`. For discrete distributions (panels 7 and 8), used `axs[i].plot(x, stats.<dist>.pmf(x, ...), "k--", lw=1.5, marker="o")`. Each line includes a legend label and a corresponding `axs[i].legend(fontsize=8)` call (or updates the existing legend).

## Key insight
The existing code already had theoretical PDF lines on panels 18–21 and 23. The pattern is consistent: compute x values spanning the distribution's support, evaluate the theoretical PDF using scipy.stats, and overlay with a black dashed line. This pattern generalizes to any distribution that scipy.stats supports.

## Verification
Ran `python3 random_dist.py` — executed without errors, saved `random_all.png` (631K). All 28 panels render correctly with theoretical curves overlaid.

## Generalizes to
The scipy.stats overlay pattern works for any distribution: compute x-range, evaluate PDF/PMF, plot with `ax.plot(x, pdf, "k--")`. This can be applied to any histogram-based distribution visualization project.