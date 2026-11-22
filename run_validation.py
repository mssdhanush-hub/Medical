"""Run the full analysis: DOE -> Gage R&R -> OQ -> PQ, and write the validation report.

Usage:  python run_validation.py
Output: reports/validation_report.md and reports/figures/*.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sealval import capability, doe, gage_rr, spc  # noqa: E402
from sealval.process_model import FACTORS, SEAL_LSL, SEAL_USL, WIDTH_LSL, WIDTH_USL, to_coded  # noqa: E402

DATA, REPORT, FIG = ROOT / "data", ROOT / "reports", ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
NAMES = list(FACTORS)

# Acceptance criteria - must match docs/protocols
OQ_MIN_PPK = 1.00
PQ_MIN_PPK = 1.33
GRR_MAX_PCT_TOL = 30.0
FIRST_WINDOW = {"temp_C": (140, 150), "pressure_psi": (45, 55), "dwell_s": (1.0, 1.4)}  # first proposal
WINDOW = {"temp_C": (143, 153), "pressure_psi": (45, 55), "dwell_s": (1.1, 1.3)}  # final, after DOE review
NOMINAL = (148, 50, 1.2)

lines = []


def md(text=""):
    lines.append(text)


def table(df, floatfmt="{:.3f}"):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    md("| " + " | ".join(map(str, df.columns)) + " |")
    md("|" + "---|" * len(df.columns))
    for row in df.itertuples(index=False):
        md("| " + " | ".join(map(str, row)) + " |")
    md()


def verdict(ok):
    return "**PASS**" if ok else "**FAIL**"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130)
    plt.close(fig)
    return f"figures/{name}"


def imr_chart(x, title, name, lsl=None, usl=None):
    lim = spc.imr_limits(x)
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    idx = np.arange(1, len(x) + 1)
    ax[0].plot(idx, x, "o-", ms=3, color="#1f4e79")
    for v, s in [(lim["centre"], "-"), (lim["ucl"], "--"), (lim["lcl"], "--")]:
        ax[0].axhline(v, ls=s, color="#c0392b" if s == "--" else "grey", lw=1)
    for v in [lsl, usl]:
        if v is not None:
            ax[0].axhline(v, color="black", lw=1.2, ls=":")
    ax[0].set_ylabel("Individual")
    ax[0].set_title(title)
    mr = np.abs(np.diff(x))
    ax[1].plot(idx[1:], mr, "o-", ms=3, color="#1f4e79")
    ax[1].axhline(lim["mr_bar"], color="grey", lw=1)
    ax[1].axhline(lim["mr_ucl"], color="#c0392b", ls="--", lw=1)
    ax[1].set_ylabel("Moving range")
    ax[1].set_xlabel("Sample")
    return save(fig, name), lim, spc.rule_violations(x, lim)


def histogram(x, title, name, lsl, usl):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(x, bins=15, color="#9dc3e6", edgecolor="#1f4e79")
    ax.axvline(lsl, color="#c0392b", lw=1.5, label="LSL / USL")
    ax.axvline(usl, color="#c0392b", lw=1.5)
    ax.set_title(title)
    ax.set_xlabel("Seal strength (lbf/in)")
    ax.legend()
    return save(fig, name)


md("# Process Validation Report - Pouch Heat-Sealing Process (VR-SEAL-001)")
md()
md("> All data is **simulated** (see `data/generate_data.py`). The method, calculations and acceptance "
   "criteria are what a real study would use.")
md()
md(f"Seal strength specification: **{SEAL_LSL} - {SEAL_USL} lbf/in**. Seal width specification: **{WIDTH_LSL} - {WIDTH_USL} mm**.")
md()

# ---------------------------------------------------------------- 1. DOE
runs = pd.read_csv(DATA / "doe_runs.csv")
coded = pd.DataFrame({f: to_coded(f, runs[f]) for f in NAMES})
res = doe.analyse(coded, runs["seal_strength_lbf_in"], NAMES)
eff = res["effects"]

md("## 1. Characterisation - Design of Experiments (DOE)")
md()
md(f"2^3 full factorial, 2 replicates + 3 centre points = {len(runs)} runs, run in random order. "
   "Each run = mean of 5 pouches.")
md()
table(pd.DataFrame({f: [f"{FACTORS[f][0]} {FACTORS[f][2]}", f"{FACTORS[f][1]} {FACTORS[f][2]}"] for f in NAMES}, index=["Low (-1)", "High (+1)"]).reset_index().rename(columns={"index": "Level"}))
show = eff[eff.term != "Intercept"][["term", "effect", "t_value", "p_value", "significant"]].sort_values("p_value")
table(show)
md(f"Model R-squared = {res['r_squared']:.3f}.")
c = res["curvature"]
md(f"Curvature check: factorial mean {c['factorial_mean']:.3f} vs centre-point mean {c['centre_mean']:.3f}, "
   f"p = {c['p_value']:.3f} -> " + ("curvature is significant; the linear model should only be trusted near the centre." if c["p_value"] < 0.05 else "no significant curvature."))
md()

# Pareto of standardised effects
sub = eff[eff.term != "Intercept"].assign(abs_t=lambda d: d.t_value.abs()).sort_values("abs_t")
from scipy import stats as _st  # noqa: E402
t_crit = _st.t.ppf(0.975, res["dof"])
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.barh(sub.term, sub.abs_t, color=["#1f4e79" if s == "yes" else "#bfbfbf" for s in sub.significant])
ax.axvline(t_crit, color="#c0392b", ls="--", label=f"t crit = {t_crit:.2f}")
ax.set_xlabel("|t| (standardised effect)")
ax.set_title("Pareto of standardised effects - seal strength")
ax.legend()
md(f"![Pareto]({save(fig, 'doe_pareto.png')})")
md()

# Main effects plot
fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
fact = runs[~np.isclose(coded[NAMES].to_numpy(), 0).all(axis=1)]
for ax, f in zip(axes, NAMES):
    m = fact.groupby(f)["seal_strength_lbf_in"].mean()
    ax.plot(m.index, m.values, "o-", color="#1f4e79")
    ax.set_xlabel(f)
axes[0].set_ylabel("Mean seal strength")
fig.suptitle("Main effects")
md(f"![Main effects]({save(fig, 'doe_main_effects.png')})")
md()

# Interaction plot temp x dwell
fig, ax = plt.subplots(figsize=(5, 3.4))
for dval, g in fact.groupby("dwell_s"):
    m = g.groupby("temp_C")["seal_strength_lbf_in"].mean()
    ax.plot(m.index, m.values, "o-", label=f"dwell {dval} s")
ax.set_xlabel("temp_C")
ax.set_ylabel("Mean seal strength")
ax.set_title("Interaction: temperature x dwell")
ax.legend()
md(f"![Interaction]({save(fig, 'doe_interaction.png')})")
md()

# Check the proposed operating window with the fitted model
md("### Choosing the operating window")
md()
import itertools  # noqa: E402
pouch_sd = np.sqrt(res["mse"] * 5)  # each DOE run is a mean of 5 pouches
margin = 3 * pouch_sd
md(f"Rule: at every corner of the window, predicted mean +/- 3 x pouch-to-pouch sigma ({pouch_sd:.3f}) must stay inside "
   f"{SEAL_LSL}-{SEAL_USL} lbf/in (a predicted Ppk of at least 1.0 at the worst case).")
md()


def check_window(window, label):
    rows = []
    for combo in itertools.product(*[window[f] for f in NAMES]):
        point = {f: float(to_coded(f, v)) for f, v in zip(NAMES, combo)}
        pred = doe.predict(res["beta"], point)
        rows.append(dict(zip(NAMES, combo), predicted=pred, low_3sd=pred - margin, high_3sd=pred + margin))
    tbl = pd.DataFrame(rows)
    ok = tbl.low_3sd.min() > SEAL_LSL and tbl.high_3sd.max() < SEAL_USL
    md(f"**{label}:** " + ", ".join(f"{f} {window[f][0]}-{window[f][1]}" for f in NAMES)
       + f" -> worst case {tbl.low_3sd.min():.2f} to {tbl.high_3sd.max():.2f} lbf/in: {verdict(ok)}")
    md()
    return tbl, ok


_, first_ok = check_window(FIRST_WINDOW, "First proposal")
corners, win_ok = check_window(WINDOW, "Final window")
table(corners)
if not first_ok:
    md("The first proposal did not leave enough margin at the low-temperature, short-dwell corner, so the window was "
       "moved up in temperature and narrowed in dwell time. The final window's worst-case corners are challenged in OQ.")
    md()

# ---------------------------------------------------------------- 2. Gage R&R
md("## 2. Test Method Validation - Gage R&R on seal width (TMV-SEAL-001)")
md()
gr = pd.read_csv(DATA / "gage_rr_seal_width.csv")
g = gage_rr.gage_rr(gr, value="seal_width_mm", tolerance=WIDTH_USL - WIDTH_LSL)
md("10 pouches x 3 operators x 3 trials, crossed study, ANOVA method.")
md()
table(g["anova"])
table(g["components"])
grr_tol = g["components"].loc[0, "pct_tolerance"]
md(f"Part x operator interaction p = {g['interaction_p']:.3f} ({'pooled into repeatability' if g['interaction_pooled'] else 'kept in model'}). "
   f"Number of distinct categories (ndc) = {g['ndc']}.")
md()
grr_ok = grr_tol <= GRR_MAX_PCT_TOL and g["ndc"] >= 5
md(f"Acceptance: %GRR of tolerance <= {GRR_MAX_PCT_TOL:.0f}% and ndc >= 5 -> {grr_tol:.1f}%, ndc {g['ndc']}: {verdict(grr_ok)} "
   f"(study variation rating: {g['verdict']}).")
md()
fig, ax = plt.subplots(figsize=(7, 3.4))
for op, d in gr.groupby("operator"):
    m = d.groupby("part")["seal_width_mm"].mean()
    ax.plot(m.index, m.values, "o-", label=f"Operator {op}")
ax.set_xlabel("Part")
ax.set_ylabel("Mean seal width (mm)")
ax.set_title("Part x operator")
ax.legend()
md(f"![Gage R&R]({save(fig, 'grr_part_operator.png')})")
md()

# ---------------------------------------------------------------- 3. OQ
md("## 3. Operational Qualification (OQ-SEAL-001)")
md()
md(f"30 pouches at each worst-case corner. Acceptance per condition: 0 out of spec, Ppk >= {OQ_MIN_PPK:.2f}.")
md()
oq = pd.read_csv(DATA / "oq_runs.csv")
oq_rows, oq_ok = [], True
for cond, d in oq.groupby("condition", sort=False):
    cap = capability.capability(d.seal_strength_lbf_in, SEAL_LSL, SEAL_USL)
    ok = cap["out_of_spec"] == 0 and cap["Ppk"] >= OQ_MIN_PPK
    oq_ok &= ok
    oq_rows.append({"condition": cond, "n": cap["n"], "mean": cap["mean"], "sd": cap["sd_overall"], "min": cap["min"], "max": cap["max"],
                    "Ppk": cap["Ppk"], "out_of_spec": cap["out_of_spec"], "result": "PASS" if ok else "FAIL"})
table(pd.DataFrame(oq_rows))
md(f"OQ overall: {verdict(oq_ok)}")
md()

# ---------------------------------------------------------------- 4. PQ
md("## 4. Performance Qualification (PQ-SEAL-001)")
md()
md(f"3 lots x 30 pouches at nominal settings ({NOMINAL[0]} C, {NOMINAL[1]} psi, {NOMINAL[2]} s). Acceptance: each lot 0 out of spec and Ppk >= {PQ_MIN_PPK}; "
   "combined data normal (Shapiro-Wilk p > 0.05). Control chart signals are not pass/fail but each one must be investigated and documented.")
md()
pq = pd.read_csv(DATA / "pq_runs.csv")
pq_rows, pq_ok = [], True
for lot, d in pq.groupby("lot"):
    cap = capability.capability(d.seal_strength_lbf_in, SEAL_LSL, SEAL_USL)
    ok = cap["out_of_spec"] == 0 and cap["Ppk"] >= PQ_MIN_PPK
    pq_ok &= ok
    pq_rows.append({"lot": lot, "n": cap["n"], "mean": cap["mean"], "sd": cap["sd_overall"], "Ppk": cap["Ppk"],
                    "Ppk_lower_95": cap["Ppk_lower_95"], "out_of_spec": cap["out_of_spec"], "result": "PASS" if ok else "FAIL"})
allcap = capability.capability(pq.seal_strength_lbf_in, SEAL_LSL, SEAL_USL)
pq_rows.append({"lot": "All lots", "n": allcap["n"], "mean": allcap["mean"], "sd": allcap["sd_overall"], "Ppk": allcap["Ppk"],
                "Ppk_lower_95": allcap["Ppk_lower_95"], "out_of_spec": allcap["out_of_spec"], "result": ""})
table(pd.DataFrame(pq_rows))
# Control chart is judged within each lot; lot-to-lot shift is checked separately with one-way ANOVA
n_signals, rule_notes = 0, []
for lot, d in pq.groupby("lot"):
    x = d.seal_strength_lbf_in.to_numpy()
    lim_l = spc.imr_limits(x)
    v = spc.rule_violations(x, lim_l)
    n_signals += len(v["rule1_beyond_3sigma"])
    rule_notes.append(f"{lot}: limits {lim_l['lcl']:.2f}-{lim_l['ucl']:.2f}, beyond 3 sigma = {len(v['rule1_beyond_3sigma'])}, "
                      f"8-same-side = {len(v['rule2_8_same_side'])}, 6-trend = {len(v['rule3_6_trending'])}")
img, lim, viol = imr_chart(pq.seal_strength_lbf_in.to_numpy(), "PQ I-MR chart - seal strength (lots 1-3 in sequence)", "pq_imr.png", SEAL_LSL, SEAL_USL)
anova_p = _st.f_oneway(*[d.seal_strength_lbf_in for _, d in pq.groupby("lot")]).pvalue
normal_ok = allcap["normality_p"] > 0.05
pq_ok &= normal_ok
md(f"Normality (Shapiro-Wilk, all lots) p = {allcap['normality_p']:.3f}: {verdict(normal_ok)}.")
md()
md(f"Control chart review (I-MR within each lot): {n_signals} point(s) beyond 3 sigma.")
md()
for note in rule_notes:
    md(f"- {note}")
md()
if n_signals:
    md("Required action: every flagged point is inside specification, and with 90 points about 0.25 false alarms are "
       "expected by chance alone. Check the batch record at that time (stoppage, roll change, alarm). If no assignable "
       "cause is found, record it as a deviation note with no impact on the PQ conclusion; if a cause is found, raise a "
       "non-conformance.")
    md()
md(f"Lot-to-lot difference (one-way ANOVA) p = {anova_p:.3f}. "
   + ("Lot means differ. Not a failure (all lots meet Ppk), but recorded as an observation: check incoming material "
      "variation between rolls and keep monitoring after release." if anova_p < 0.05 else "No significant difference between lots."))
md()
md(f"![PQ I-MR]({img})")
md()
md(f"![PQ histogram]({histogram(pq.seal_strength_lbf_in, 'PQ seal strength - all lots', 'pq_histogram.png', SEAL_LSL, SEAL_USL)})")
md()
md(f"PQ overall: {verdict(pq_ok)}")
md()

# ---------------------------------------------------------------- Summary
md("## 5. Conclusion")
md()
summary = pd.DataFrame([
    {"stage": "DOE / operating window", "result": "PASS" if win_ok else "FAIL"},
    {"stage": "Test method (Gage R&R)", "result": "PASS" if grr_ok else "FAIL"},
    {"stage": "OQ", "result": "PASS" if oq_ok else "FAIL"},
    {"stage": "PQ", "result": "PASS" if pq_ok else "FAIL"},
])
table(summary)
all_ok = win_ok and grr_ok and oq_ok and pq_ok
md("The sealing process is **validated** for the operating window " + ", ".join(f"{f} {WINDOW[f][0]}-{WINDOW[f][1]}" for f in NAMES) + "." if all_ok
   else "The process is **not** validated. Raise a non-conformance, investigate root cause and repeat the failed stage.")
md()
md("Recommended ongoing controls: I-MR chart on 5 pouches per shift, re-validation on sealer repair, "
   "material change or any result outside control limits (see docs/pFMEA.csv).")

(REPORT / "validation_report.md").write_text("\n".join(lines))
print("\n".join(f"{r.stage}: {r.result}" for r in summary.itertuples()))
print("Report written to", REPORT / "validation_report.md")
