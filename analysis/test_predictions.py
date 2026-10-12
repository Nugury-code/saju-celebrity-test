"""Step 5: occupation-specific predictions P1-P3 (P4 needs a fame column).
Predictions and the corrected alpha (0.005) were fixed in the progress log before running.

P1 코미디언·사회자 -> A1 식상 (month/day branch main qi is 식신/상관 vs day stem)
P2 배우 계열       -> B  도화살
P3 가수 계열       -> B  도화살
Baseline: per birth year, share of calendar days on which the rule holds.
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
DOHWA = {}
for grp, tgt in (("인오술", "묘"), ("사유축", "오"), ("신자진", "유"), ("해묘미", "자")):
    for b in grp:
        DOHWA[b] = tgt
JIE_ODD = set(range(1, 24, 2))

COMEDY_HOST = {"코미디언", "owarai tarento", "TV 사회자", "game show host", "쇼핑 호스트"}
ACTOR = {"배우", "영화 배우", "텔레비전 배우", "연극 배우", "뮤지컬 배우", "성우",
         "일본의 성우", "스턴트 배우", "포르노 배우", "성인용 비디오 배우", "video game actor"}
SINGER = {"가수", "팝 가수", "싱어송라이터", "래퍼", "오페라 가수", "vocalist", "성악가",
          "bass singer", "chanteuse", "child singer", "jazz singer"}


def is_siksang(ilgan, branch):
    return ELEM[BONGI[branch]] == (ELEM[ilgan] + 1) % 5


def rules(year_p, month_p, day_p):
    ilgan, dj, mj, yj = day_p[0], day_p[1], month_p[1], year_p[1]
    a1 = is_siksang(ilgan, mj) or is_siksang(ilgan, dj)
    b = (DOHWA[yj] in {yj, mj, dj}) or (DOHWA[dj] in {yj, mj, dj})
    return a1, b


cache = {}


def year_table(y):
    if y in cache:
        return cache[y]
    n = 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
    base = date(y, 1, 1).toordinal()
    rows = []
    for k in range(n):
        dt = date.fromordinal(base + k)
        if DROP and is_term_day(dt):
            continue
        d = sxtwl.fromSolar(dt.year, dt.month, dt.day)
        yg, mg, dg = YG(dt), MG(dt), d.getDayGZ()
        rows.append(rules(GAN[yg.tg] + JI[yg.dz], GAN[mg.tg] + JI[mg.dz], GAN[dg.tg] + JI[dg.dz]))
    cache[y] = np.array(rows, dtype=bool)
    return cache[y]


p = pd.read_csv(SRC)
p["occ_list"] = p.occs.str.split("|")


def has_any(S):
    return p.occ_list.apply(lambda o: any(x in S for x in o))


tests = [
    ("P1 코미디언·사회자 → 식상(A1)", has_any(COMEDY_HOST), 0),
    ("P2 배우 계열 → 도화살(B)", has_any(ACTOR), 1),
    ("P3 가수 계열 → 도화살(B)", has_any(SINGER), 1),
]
lines = ["| 예측 | N | 해당 % | 달력 기대 % | z | 양쪽 p | 기준(0.005) | 방향 |", "|---|---|---|---|---|---|---|---|"]
for name, mask, j in tests:
    sub = p[mask]
    hits = exp = var = 0.0
    for r in sub.itertuples():
        hit = rules(r.year_pillar, r.month_pillar, r.day_pillar)[j]
        pr = year_table(int(r.year))[:, j].mean()
        hits += hit
        exp += pr
        var += pr * (1 - pr)
    n = len(sub)
    z = (hits - exp) / var ** 0.5
    pv = 2 * stats.norm.sf(abs(z))
    lines.append(f"| {name} | {n:,} | {100*hits/n:.1f} | {100*exp/n:.1f} | {z:.2f} | {pv:.3f} | "
                 f"{'통과' if pv < 0.005 else '통과 못 함'} | {'예측대로(높음)' if z > 0 else '예측과 반대(낮음)'} |")
open(OUT_MD, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
