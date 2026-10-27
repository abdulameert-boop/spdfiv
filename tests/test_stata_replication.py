"""Cross-software test against the published Stata results.

Runs only if the environment variable SPDFIV_JSS_DATA points to the folder
with the replication materials of Kripfganz and Sarafidis (2025, Journal of
Statistical Software 113(6)): v113i06.dta and W.csv.
"""

import os
import unittest
import warnings

import numpy as np
import pandas as pd

import spdfiv as sd

DATA = os.environ.get("SPDFIV_JSS_DATA")

TABLE1 = {"W.NPL": .3943206, "L1.NPL": .2898521, "INEFF": .4473777, "CAR": .0305078,
          "SIZE": .2225966, "BUFFER": -.0545049, "PROFIT": -.0053351, "QUALITY": .1830412,
          "LIQUIDITY": 2.452391}
TABLE1_SE = {"W.NPL": .0848856, "L1.NPL": .0543794, "INEFF": .1045636, "CAR": .0057852,
             "SIZE": .0941614, "BUFFER": .0118678, "PROFIT": .0018411, "QUALITY": .0307657,
             "LIQUIDITY": .2696471}
TABLE5 = {"W.NPL": .031593, "L1.NPL": .3005247, "INEFF": .7587664, "CAR": .218054,
          "SIZE": 2.004026, "BUFFER": -.3763774, "PROFIT": -.0179663, "QUALITY": .2872525,
          "LIQUIDITY": 6.330179}
TABLE5_SE = {"W.NPL": .0511028, "L1.NPL": .0148501, "INEFF": .1583511, "CAR": .0262755,
             "SIZE": .3385335, "BUFFER": .0420252, "PROFIT": .005944, "QUALITY": .1386973,
             "LIQUIDITY": .5059499}
X = ["INEFF", "CAR", "SIZE", "BUFFER", "PROFIT", "QUALITY", "LIQUIDITY"]
IV = ["INTEREST", "CAR", "SIZE", "BUFFER", "PROFIT", "QUALITY", "LIQUIDITY"]


@unittest.skipUnless(DATA, "set SPDFIV_JSS_DATA to run the Stata cross-check")
class TestAgainstStata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = pd.read_stata(os.path.join(DATA, "v113i06.dta"))
        W = pd.read_csv(os.path.join(DATA, "W.csv"), header=None).to_numpy()
        cls.kw = dict(W_ids=sorted(d["ID"].unique()), splag=True, tlags=1, iv=IV,
                      iv_lags=1, iv_splags=True, std=True)
        cls.d, cls.W = d, W

    def test_table1(self):
        r = sd.fit(self.d, "NPL", X, "ID", "TIME", self.W, **self.kw)
        for k in TABLE1:      # printed with 7 significant digits
            self.assertAlmostEqual(r.params[k], TABLE1[k], delta=2e-6 * max(1, abs(TABLE1[k])))
            self.assertAlmostEqual(r.bse[k], TABLE1_SE[k], delta=2e-6 * max(1, TABLE1_SE[k]))
        self.assertAlmostEqual(r.J, 18.8250, places=3)
        self.assertEqual(r.factors_x, [2, 2])
        self.assertEqual(r.factors_u, 1)

    def test_table5(self):
        r = sd.fit(self.d, "NPL", X, "ID", "TIME", self.W, estimator="mg", **self.kw)
        for k in TABLE5:
            self.assertAlmostEqual(r.params[k], TABLE5[k], delta=2e-6 * max(1, abs(TABLE5[k])))
            self.assertAlmostEqual(r.bse[k], TABLE5_SE[k], delta=2e-6 * max(1, TABLE5_SE[k]))


if __name__ == "__main__":
    unittest.main()
