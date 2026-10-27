# Replication materials

Run from the project root with the package installed (`pip install .`) or
with `PYTHONPATH=.:replication`.

| Script | Paper section | Data needed |
|---|---|---|
| `illustration.py` | 4 (Illustration) | none (simulated) |
| `monte_carlo.py --reps 1000 --workers 2` | 5 (Monte Carlo) | none (simulated) |
| `make_tables.py` | 5 and 7 (writes `paper/tables/*.tex`) | output of `monte_carlo.py` |
| `bank_replication.py v113i06.dta W.csv` | 6 (bank data) | replication materials of Kripfganz and Sarafidis (2025) |
| `run_all.sh v113i06.dta W.csv` | everything, then rebuilds the PDF | as above |

The bank data and weights matrix are part of the replication materials of
Kripfganz, S. and Sarafidis, V. (2025), Journal of Statistical Software 113(6),
https://doi.org/10.18637/jss.v113.i06 ("Replication materials" on the article
page). They are not redistributed here.

All text outputs are written to `replication/results/`; LaTeX tables and the
numbers quoted in the text are written to `paper/tables/`, from which the
paper picks them up on the next LaTeX run.

The Monte Carlo study takes about 30 minutes on two cores.
