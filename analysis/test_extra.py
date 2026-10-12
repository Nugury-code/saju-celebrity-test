"""Step 4 (exploratory): 재성 and 홍염살, rules fixed in the progress log before running.

R1 재성 있음 : an element controlled by the day stem appears among year stem, month stem
               and the main qi of year/month/day branch (5 characters; hour pillar unknown).
R2 홍염살    : the 홍염 branch of the day stem is among {년지, 월지, 일지}.
Same calendar baseline as test_pattern.py (per-year share of days on which the rule holds).
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

SRC, OUT_MD = sys.argv[1], sys.argv[2]
GAN = list("갑을병정무기경신임계")
JI = list("자축인묘진사오미신유술해")
ELEM = {"갑": 0, "을": 0, "병": 1, "정": 1, "무": 2, "기": 2, "경": 3, "신": 3, "임": 4, "계": 4}
BONGI = dict(zip("자축인묘진사오미신유술해", "계기갑을무병정기경신무임"))
HONGYEOM = {"갑": "오", "을": "오", "병": "인", "정": "미", "무": "진",
            "기": "진", "경": "술", "신": "유", "임": "자", "계": "신"}
JIE_ODD = set(range(1, 24, 2))


def rules(year_p, month_p, day_p):
    ilgan = day_p[0]
    target = (ELEM[ilgan] + 2) % 5  # element controlled by the day stem
    chars = [year_p[0], month_p[0], BONGI[year_p[1]], BONGI[month_p[1]], BONGI[day_p[1]]]
    r1 = any(ELEM[c] == target for c in chars)
    r2 = HONGYEOM[ilgan] in {year_p[1], month_p[1], day_p[1]}
    return r1, r2


cache = {}


def year_table(y):
    if y in cache:
        return cache[y]
    n = 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
    base = date(y, 1, 1).toordinal()
    jie = {}
    for k in range(-1, n + 1):
        dt = date.fromordinal(base + k)
        d = sxtwl.fromSolar(dt.year, dt.month, dt.day)
        jie[k] = bool(d.hasJieQi() and d.getJieQi() in JIE_ODD)
    rows = []
    for k in range(n):
        dt = date.fromordinal(base + k)
        if DROP and is_term_day(dt):
            continue
        d = sxtwl.fromSolar(dt.year, dt.month, dt.day)
        yg, mg, dg = YG(dt), MG(dt), d.getDayGZ()
        r1, r2 = rules(GAN[yg.tg] + JI[yg.dz], GAN[mg.tg] + JI[mg.dz], GAN[dg.tg] + JI[dg.dz])
        rows.append((r1, r2, jie[k - 1] or jie[k] or jie[k + 1]))
    cache[y] = np.array(rows, dtype=bool)
    return cache[y]


def evaluate(sub, drop_near):
    out = {}
    for j, name in enumerate(["R1 재성 있음", "R2 홍염살"]):
        hits = exp = var = 0.0
        for r in sub.itertuples():
            hit = rules(r.year_pillar, r.month_pillar, r.day_pillar)[j]
            tab = year_table(int(r.year))
            if drop_near:
                tab = tab[~tab[:, 2]]
            pr = tab[:, j].mean()
            hits += hit
            exp += pr
            var += pr * (1 - pr)
        n = len(sub)
        z = (hits - exp) / var ** 0.5
        out[name] = (n, 100 * hits / n, 100 * exp / n, z, 2 * stats.norm.sf(abs(z)))
    return out


p = pd.read_csv(SRC)
lines = ["| 분석 | 조건 | N | 연예인 해당 % | 달력 기대 % (= 일반 날짜) | z | p | 보정 기준(0.01) |", "|---|---|---|---|---|---|---|---|"]
for name, sub, drop in [("주 분석: 전체", p, False), ("민감도: 절기 근처 날 제외", p[~p.near_jie_day], True)]:
    for rule, (n, o, e, z, pv) in evaluate(sub, drop).items():
        lines.append(f"| {name} | {rule} | {n:,} | {o:.1f} | {e:.1f} | {z:.2f} | {pv:.3f} | {'통과' if pv < 0.01 else '통과 못 함'} |")
open(OUT_MD, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
