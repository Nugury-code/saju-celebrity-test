"""Step 6: P4 - are the most famous entertainers more likely than the rest to have both
식상 (A1) and 도화살 (B)?  Fame = Wikidata sitelinks count. Top group = sitelinks >= the
90th percentile among the 12,265 entertainers (ties included). People missing from the
fame file are treated as sitelinks < 3. Rule and alpha (0.005) fixed in the progress log.

Test: excess = observed - calendar-expected rate of C = A1 and B. Compare the excess of
the top group with the excess of the rest (difference z), and the top group alone vs baseline.
"""
import sys
from datetime import date

import numpy as np
import pandas as pd
import sxtwl
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from saju_kst import YG, MG  # KST 절기 (see saju_kst.py)
from saju_kst import is_term_day
import os
DROP = os.environ.get('SAJU_DROP_TERM_DAYS') == '1'  # main analysis: leave out 절-day births and the same days of the baseline calendar
from scipy import stats

PEOPLE, FAME, OUT_MD = sys.argv[1], sys.argv[2], sys.argv[3]
GAN = list("갑을병정무기경신임계")
JI = list("자축인묘진사오미신유술해")
ELEM = {"갑": 0, "을": 0, "병": 1, "정": 1, "무": 2, "기": 2, "경": 3, "신": 3, "임": 4, "계": 4}
BONGI = dict(zip("자축인묘진사오미신유술해", "계기갑을무병정기경신무임"))
DOHWA = {}
for grp, tgt in (("인오술", "묘"), ("사유축", "오"), ("신자진", "유"), ("해묘미", "자")):
    for b in grp:
        DOHWA[b] = tgt


def rule_c(year_p, month_p, day_p):
    ilgan, dj, mj, yj = day_p[0], day_p[1], month_p[1], year_p[1]
    sik = lambda br: ELEM[BONGI[br]] == (ELEM[ilgan] + 1) % 5
    a1 = sik(mj) or sik(dj)
    b = (DOHWA[yj] in {yj, mj, dj}) or (DOHWA[dj] in {yj, mj, dj})
    return a1 and b


cache = {}


def year_p_c(y):
    if y not in cache:
        n = 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
        base = date(y, 1, 1).toordinal()
        v = []
        for k in range(n):
            dt = date.fromordinal(base + k)
            if DROP and is_term_day(dt):
                continue
            d = sxtwl.fromSolar(dt.year, dt.month, dt.day)
            yg, mg, dg = YG(dt), MG(dt), d.getDayGZ()
            v.append(rule_c(GAN[yg.tg] + JI[yg.dz], GAN[mg.tg] + JI[mg.dz], GAN[dg.tg] + JI[dg.dz]))
        cache[y] = float(np.mean(v))
    return cache[y]


people = pd.read_csv(PEOPLE)
fame = pd.read_csv(FAME)
print("fame file rows:", len(fame), "| columns:", list(fame.columns))
fame = fame.drop_duplicates("p")
people = people.merge(fame[["p", "sl"]], on="p", how="left")
matched = int(people.sl.notna().sum())
people["sl"] = people.sl.fillna(0)  # not in file => fewer than 3 language editions
thr = float(np.percentile(people.sl, 90))
print(f"matched in fame file: {matched} of {len(people)} | 90th percentile of sitelinks: {thr}")
if thr < 3:
    sys.exit("90th percentile is below the file's cutoff (3): need a lower cutoff, stopping.")
people["top"] = people.sl >= thr
print("top group size:", int(people.top.sum()), "| rest:", int((~people.top).sum()))
print(people.sl.describe(percentiles=[.5, .75, .9, .95]).round(1).to_string())

people["hit"] = [rule_c(r.year_pillar, r.month_pillar, r.day_pillar) for r in people.itertuples()]
people["pr"] = [year_p_c(int(y)) for y in people.year]


def group(df):
    n = len(df)
    obs = df.hit.mean()
    exp = df.pr.mean()
    var = (df.pr * (1 - df.pr)).sum() / n ** 2
    return n, obs, exp, var


nt, ot, et, vt = group(people[people.top])
nr, orr, er, vr = group(people[~people.top])
z_top = (ot - et) / vt ** 0.5
diff = (ot - et) - (orr - er)
z_diff = diff / (vt + vr) ** 0.5
p_diff = 2 * stats.norm.sf(abs(z_diff))
p_top = 2 * stats.norm.sf(abs(z_top))

lines = [
    "| 집단 | N | C(식상+도화) 해당 % | 달력 기대 % |",
    "|---|---|---|---|",
    f"| 유명도 상위 10% (언어판 {thr:.0f}개 이상) | {nt:,} | {100*ot:.1f} | {100*et:.1f} |",
    f"| 나머지 | {nr:,} | {100*orr:.1f} | {100*er:.1f} |",
    "",
    f"- 상위 집단만 기준과 비교: z={z_top:.2f}, 양쪽 p={p_top:.3f}",
    f"- P4 본 검정(상위 초과분 − 나머지 초과분 = {100*diff:+.1f}%p): z={z_diff:.2f}, 양쪽 p={p_diff:.3f}, "
    f"기준 0.005 {'통과' if p_diff < 0.005 else '통과 못 함'}, 방향 {'예측대로(상위가 높음)' if diff > 0 else '예측과 반대'}",
]
open(OUT_MD, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
