"""Calendar-baseline share of each candidate actor condition (descriptive only; no people).
Weighted by the birth-year counts of the main sample, term days left out like the main analysis.
usage: python3 candidates_baseline.py data/people_main.csv
"""
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import sxtwl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saju_kst import YG, MG, is_term_day

GAN = list("갑을병정무기경신임계")
JI = list("자축인묘진사오미신유술해")
ELEM = {g: i // 2 for i, g in enumerate(GAN)}
YANG = {g: (i % 2 == 0) for i, g in enumerate(GAN)}
BONGI = dict(zip("자축인묘진사오미신유술해", "계기갑을무병정기경신무임"))
DOHWA = {}
for grp, tgt in (("인오술", "묘"), ("사유축", "오"), ("신자진", "유"), ("해묘미", "자")):
    for b in grp:
        DOHWA[b] = tgt


def god(ilgan, other):
    """ten-god name of stem `other` relative to day stem `ilgan`"""
    d = (ELEM[other] - ELEM[ilgan]) % 5
    same = YANG[other] == YANG[ilgan]
    return {0: ("비견", "겁재"), 1: ("식신", "상관"), 2: ("편재", "정재"),
            3: ("편관", "정관"), 4: ("편인", "정인")}[d][0 if same else 1]


NAMES = ["식신·상관", "상관", "편인", "겁재", "도화살", "일간 금·수"]


def conds(yp, mp, dp):
    il = dp[0]
    gs = {god(il, BONGI[mp[1]]), god(il, BONGI[dp[1]])}  # 월지·일지 본기
    br = {yp[1], mp[1], dp[1]}
    dohwa = (DOHWA[yp[1]] in br) or (DOHWA[dp[1]] in br)
    return [bool(gs & {"식신", "상관"}), "상관" in gs, "편인" in gs, "겁재" in gs,
            dohwa, il in "경신임계"]


p = pd.read_csv(sys.argv[1])
yc = p["year"].astype(int).value_counts().to_dict()
N = sum(yc.values())
acc = np.zeros(len(NAMES))
anyc = 0.0
for y, c in yc.items():
    n = 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
    base = date(y, 1, 1).toordinal()
    rows = []
    for k in range(n):
        dt = date.fromordinal(base + k)
        if is_term_day(dt):
            continue
        d = sxtwl.fromSolar(dt.year, dt.month, dt.day)
        yg, mg, dg = YG(dt), MG(dt), d.getDayGZ()
        rows.append(conds(GAN[yg.tg] + JI[yg.dz], GAN[mg.tg] + JI[mg.dz], GAN[dg.tg] + JI[dg.dz]))
    a = np.array(rows, dtype=float)
    acc += c * a.mean(0)
    anyc += c * (a.max(1)).mean()
print("N", N)
for nm, v in zip(NAMES, acc / N):
    print(f"{nm}: {100 * v:.1f}%")
print(f"하나라도 해당: {100 * anyc / N:.1f}%")
