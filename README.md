# Medical Pouch Heat-Seal Process Validation (DOE → TMV → OQ → PQ)

A complete process-validation workflow for a medical-device pouch heat-sealing process, the kind of work a
Manufacturing Engineer does before a process is released to production in a regulated clean room.

The Python code does the statistics. The `docs/` folder holds the paperwork a real validation needs:
validation plan, protocols with acceptance criteria written **before** testing, pFMEA and a work instruction.

> **The data is simulated.** I don't have access to a real sealer, so `data/generate_data.py` creates realistic
> data from a process model with known behaviour. The methods, acceptance criteria and documents are the real
> deliverable. Replace the CSV files with real measurements and the analysis runs unchanged.

## What it shows

| Step | Question answered | Method | Output |
|---|---|---|---|
| Risk analysis | What can go wrong, and what needs controlling? | pFMEA with RPN scoring | `docs/pFMEA.csv` |
| Characterisation | Which settings drive seal strength? Where should the process run? | 2³ full factorial DOE with replicates and centre points | effects table, Pareto, interaction plot, operating window |
| Test method validation | Is the measurement good enough to trust? | Crossed Gage R&R (ANOVA method) | %GRR, ndc |
| OQ | Does the process work at the edges of its window? | 30 pouches at worst-case low and high corners | Ppk per condition |
| PQ | Is the process repeatable in real production? | 3 lots × 30 at nominal, different material rolls | Ppk with 95% lower bound, normality, I-MR charts, lot-to-lot ANOVA |

## Key findings (simulated study)

- **Temperature has the largest effect** on seal strength, then dwell time and pressure. There is also a
  temperature × dwell interaction: longer dwell matters more at high temperature.
- **The first proposed window failed.** At 140 °C / 1.0 s the predicted worst case dropped below the lower
  spec limit once normal pouch-to-pouch variation was added. The window was moved to **143–153 °C,
  45–55 psi, 1.1–1.3 s** *before* OQ, so the OQ was not wasted.
- **Gage R&R: 10.9 % of tolerance, ndc = 11.** Good enough to accept or reject product.
- **OQ passed** at both worst-case corners (Ppk 2.07 low, 1.23 high).
- **PQ passed** (Ppk 1.69–2.46 per lot). Two things were flagged for follow-up rather than ignored:
  one control-chart point beyond 3σ (inside spec), and a statistically significant shift between material
  rolls (ANOVA p = 0.004). The pFMEA action for that is to seal-test the first pouches of every new roll.

Full report: [`reports/validation_report.md`](reports/validation_report.md)

<p>
<img src="reports/figures/doe_pareto.png" width="48%">
<img src="reports/figures/doe_interaction.png" width="36%">
</p>
<img src="reports/figures/pq_imr.png" width="70%">

## Run it

```bash
pip install -r requirements.txt
python data/generate_data.py      # optional - regenerate the simulated data
python run_validation.py          # writes reports/validation_report.md and figures
python -m pytest -q               # unit tests for the statistics
```

## Repository layout

```
data/            simulated study data + generator
docs/            validation plan, TMV / OQ / PQ protocols, pFMEA, peel-test work instruction
src/sealval/     doe.py, capability.py, gage_rr.py, spc.py, process_model.py
reports/         generated validation report and charts
tests/           unit tests checking the formulas against hand calculations
```

## Statistics used
- **DOE:** least-squares fit on coded factors, effect = 2 × coefficient, t-tests, centre-point curvature test
- **Capability:** Cp/Cpk (short-term sigma from moving range), Pp/Ppk (overall sigma), Bissell lower 95% bound on Ppk, Shapiro-Wilk normality
- **Gage R&R:** two-way crossed ANOVA, interaction pooled when p > 0.25, %contribution, %study variation, %tolerance, ndc
- **SPC:** I-MR charts, Western Electric rules 1–3

## Standards this follows in spirit
ISO 11607-2 (validation of packaging forming and sealing processes), ISO 13485 §7.5.6 (process validation),
ISO 14971 (risk management), GHTF/SG3/N99-10 (process validation guidance), ASTM F88 (seal strength).
