"""Sensitivity bound: how far could an unmodelled birth seasonality move the
calendar-based expected shares of A1 / B / C / R1 / R2?

NOT a correction (no real monthly-birth data used). Hypothetical seasonal weights
w(m) = 1 + amp*cos(2*pi*(m - peak)/12), amp in {0.05, 0.10}, every peak month 1..12.
For each scenario: expected % of A1/B/C over the main sample (same rules and
term-day handling as test_pattern.py with SAJU_DROP_TERM_DAYS=1), and the shift
relative to the uniform-calendar expectation, in percentage points and in units of the
observed-sample standard error.

usage: python3 sensitivity_seasonality.py data/people_main.csv OUT.md
"""
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import sxtwl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saju_kst import YG, MG, is_term_day

SRC, OUT = sys.argv[1], sys.argv[2]
GAN = list("갑을병정무기경신임계")
JI = list("자축인묘진사오미신유술해")
ELEM = {"갑": 0, "을": 0, "병": 1, "정": 1, "무": 2, "기": 2, "경": 3, "신": 3, "임": 4, "계": 4}
BONGI = dict(zip("자축인묘진사오미신유술해", "계기갑을무병정기경신무임"))
HONGYEOM = {"갑": "오", "을": "오", "병": "인", "정": "미", "무": "진",
            "기": "진", "경": "술", "신": "유", "임": "자", "계": "신"}
DOHWA = {}
for grp, tgt in (("인오술", "묘"), ("사유축", "오"), ("신자진", "유"), ("해묘미", "자")):
    for b in grp:
        DOHWA[b] = tgt


def is_siksang(ilgan, branch):
    return ELEM[BONGI[branch]] == (ELEM[ilgan] + 1) % 5


def rules(yp, mp, dp):
    ilgan, dj, mj, yj = dp[0], dp[1], mp[1], yp[1]
    a1 = is_siksang(ilgan, mj) or is_siksang(ilgan, dj)
    br = {yj, mj, dj}
    b = (DOHWA[yj] in br) or (DOHWA[dj] in br)
    target = (ELEM[ilgan] + 2) % 5
    chars = [yp[0], mp[0], BONGI[yj], BONGI[mj], BONGI[dj]]
    r1 = any(ELEM[c] == target for c in chars)
    r2 = HONGYEOM[ilgan] in {yj, mj, dj}
    return a1, b, a1 and b, r1, r2


NAMES = ["A1 식상", "B 도화살", "C 둘 다", "R1 재성", "R2 홍염살"]
cache = {}


def year_arrays(y):
    if y in cache:
        return cache[y]
    n = 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
    base = date(y, 1, 1).toordinal()
    flags, months = [], []
    for k in range(n):
        dt = date.fromordinal(base + k)
        if is_term_day(dt):
            continue
        d = sxtwl.fromSolar(dt.year, dt.month, dt.day)
        yg, mg, dg = YG(dt), MG(dt), d.getDayGZ()
        yp, mp, dp = GAN[yg.tg] + JI[yg.dz], GAN[mg.tg] + JI[mg.dz], GAN[dg.tg] + JI[dg.dz]
        flags.append(rules(yp, mp, dp))
        months.append(dt.month)
    cache[y] = (np.array(flags, dtype=float), np.array(months))
    return cache[y]


p = pd.read_csv(SRC)
years = p["year"].astype(int).values
N = len(years)
obs = np.zeros(5)
for r in p.itertuples():
    obs += np.array(rules(r.year_pillar, r.month_pillar, r.day_pillar), dtype=float)
obs_pct = 100 * obs / N

yc = pd.Series(years).value_counts().to_dict()


def expected(weights):
    """expected % under month weights (len 12, index month-1)"""
    e = np.zeros(5)
    for y, c in yc.items():
        f, m = year_arrays(y)
        w = weights[m - 1]
        e += c * (f * w[:, None]).sum(0) / w.sum()
    return 100 * e / N


uni = expected(np.ones(12))
se = np.sqrt(uni / 100 * (1 - uni / 100) / N) * 100

lines = ["# 계절성 민감도 상한 (가상 가중치, 보정이 아님)", "",
         f"주 분석 표본 N={N}. 균등 달력 기대 / 관측 / 1 표준오차(%p):", "",
         "| 조건 | 관측 % | 균등 기대 % | SE %p |", "|---|---|---|---|"]
for j, nm in enumerate(NAMES):
    lines.append(f"| {nm} | {obs_pct[j]:.2f} | {uni[j]:.2f} | {se[j]:.2f} |")
lines += ["", "가상 계절 가중치 w(m)=1+amp·cos(2π(m−peak)/12)에서 기대 %의 균등 대비 최대 이동(%p)과 그 이동의 SE 배수:", "",
          "| amp | 조건 | 최대 이동 %p | SE 배수 | 이동이 최대인 peak 달 |", "|---|---|---|---|---|"]
for amp in (0.05, 0.10):
    sh = np.zeros((12, 5))
    for pk in range(1, 13):
        w = 1 + amp * np.cos(2 * np.pi * (np.arange(1, 13) - pk) / 12)
        sh[pk - 1] = expected(w) - uni
    for j, nm in enumerate(NAMES):
        k = int(np.argmax(np.abs(sh[:, j])))
        lines.append(f"| {amp:.2f} | {nm} | {sh[k, j]:+.2f} | {abs(sh[k, j]) / se[j]:.1f} | {k + 1}월 |")
open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
