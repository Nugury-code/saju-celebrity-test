"""P6 (fixed in the progress log 2026-10-12 10:56 BEFORE running): Korean actors who are also
models, or Miss Korea participants/winners, have a higher 도화살 (B) share than the same-birth-year
calendar baseline.  Same B definition, baseline, term-day handling and z-test as test_pattern.py.
Only B is computed here (no other condition), by design.

usage: python3 test_looks.py data/people_main.csv data/looks_group_raw.txt OUT.md
"""
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import sxtwl
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saju_kst import YG, MG, is_term_day

SRC, GRP, OUT = sys.argv[1:4]
GAN = list("갑을병정무기경신임계")
JI = list("자축인묘진사오미신유술해")
DOHWA = {}
for grp, tgt in (("인오술", "묘"), ("사유축", "오"), ("신자진", "유"), ("해묘미", "자")):
    for b in grp:
        DOHWA[b] = tgt


def is_b(yp, mp, dp):
    br = {yp[1], mp[1], dp[1]}
    return (DOHWA[yp[1]] in br) or (DOHWA[dp[1]] in br)


cache = {}


def base_b(y):
    if y in cache:
        return cache[y]
    n = 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
    base = date(y, 1, 1).toordinal()
    v = []
    for k in range(n):
        dt = date.fromordinal(base + k)
        if is_term_day(dt):
            continue
        d = sxtwl.fromSolar(dt.year, dt.month, dt.day)
        yg, mg, dg = YG(dt), MG(dt), d.getDayGZ()
        v.append(is_b(GAN[yg.tg] + JI[yg.dz], GAN[mg.tg] + JI[mg.dz], GAN[dg.tg] + JI[dg.dz]))
    cache[y] = float(np.mean(v))
    return cache[y]


raw = open(GRP, encoding="utf-8").read().strip().split(",")
src = {}
for t in raw:
    q, s = t.split(":")
    src.setdefault(q, set()).add(s)

p = pd.read_csv(SRC)
p["qid"] = p["p"].str.rsplit("/", n=1).str[-1]
in_list = p[p["qid"].isin(src)].copy()


def test(sub, name):
    obs = exp = var = 0.0
    for r in sub.itertuples():
        hit = is_b(r.year_pillar, r.month_pillar, r.day_pillar)
        pr = base_b(int(r.year))
        obs += hit
        exp += pr
        var += pr * (1 - pr)
    n = len(sub)
    z = (obs - exp) / var ** 0.5
    return (name, n, int(obs), 100 * obs / n, 100 * exp / n, z, 2 * stats.norm.sf(abs(z)), stats.norm.sf(z))


rows = [test(in_list, "X 전체 (배우+모델 또는 미스코리아)")]
m = in_list[in_list["qid"].map(lambda q: "m" in src[q])]
pg = in_list[in_list["qid"].map(lambda q: "p" in src[q])]
rows.append(test(m, "참고: 배우+모델"))
rows.append(test(pg, "참고: 미스코리아 참가·우승"))

lines = ["# P6 외모 대리 기준 집단의 도화살(B) 비율", "",
         f"위키데이터 목록 {len(src)}명 중 주 분석 표본(N={len(p)})과 ID가 겹친 사람: {len(in_list)}명 "
         f"(표본에 없는 사람 {len(src) - len(in_list)}명: 빠진 이유는 확인하지 않음).", "",
         "| 집단 | N | B 해당 | 관측 % | 달력 기대 % | z | p(양측) | p(한쪽, 예측 방향) |", "|---|---|---|---|---|---|---|---|"]
for n_, n, o, op, ep, z, p2, p1 in rows:
    lines.append(f"| {n_} | {n} | {o} | {op:.1f} | {ep:.1f} | {z:.2f} | {p2:.3f} | {p1:.3f} |")
lines += ["", "검정은 첫 행(X 전체)만 해당하고 기준은 α=0.05/12≈0.0042(양측). 나머지 두 행은 참고용."]
open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
