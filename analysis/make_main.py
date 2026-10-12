"""Main analysis sample (decided 2026-10-12 after the KASI check): leave out
 (a) people born before 1921 (outside KASI's official 절기 table 1920-2100, cannot be checked), and
 (b) people born on a day a 절(節) falls (KST), because the month/year pillar then depends on the
     unknown birth time.
Input data/people_clean.csv -> data/people_main.csv (+ prints the counts)."""
import os, sys
from datetime import date
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saju_kst import is_term_day

p = pd.read_csv("data/people_clean.csv")
d = pd.to_datetime(p.dob)
p["pre1921"] = d.dt.year < 1921
p["term_day"] = [is_term_day(x.date()) for x in d]
keep = ~p.pre1921 & ~p.term_day
print("total", len(p), "| born before 1921:", int(p.pre1921.sum()),
      "| 절 day (1921+):", int((p.term_day & ~p.pre1921).sum()),
      "| 절 day (all years):", int(p.term_day.sum()), "| kept:", int(keep.sum()))
p[keep].drop(columns=["pre1921", "term_day"]).to_csv("data/people_main.csv", index=False)
