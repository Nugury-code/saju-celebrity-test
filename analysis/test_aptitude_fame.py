"""1차: 배우 중 상관 보유자(월지·일지 본기 중 하나가 일간 기준 상관)의 유명도(sl) vs 비보유자.
2차 탐색: 식신·상관, 편인, 도화살.  정의는 progress log 2026-10-12 11:35에 계산 전 고정.
usage: python3 -I analysis/test_aptitude_fame.py data/people_main.csv data/fame_raw.csv OUT.md
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

PEOPLE, FAME, OUT = sys.argv[1:4]
GAN = list("갑을병정무기경신임계")
ELEM = {g: i // 2 for i, g in enumerate(GAN)}
YANG = {g: (i % 2 == 0) for i, g in enumerate(GAN)}
BONGI = dict(zip("자축인묘진사오미신유술해", "계기갑을무병정기경신무임"))
DOHWA = {}
for grp, tgt in (("인오술", "묘"), ("사유축", "오"), ("신자진", "유"), ("해묘미", "자")):
    for b in grp:
        DOHWA[b] = tgt
ACTOR = {"배우", "영화 배우", "텔레비전 배우", "연극 배우", "뮤지컬 배우", "성우",
         "일본의 성우", "스턴트 배우", "포르노 배우", "성인용 비디오 배우", "video game actor"}


def god(il, o):
    d = (ELEM[o] - ELEM[il]) % 5
    same = YANG[o] == YANG[il]
    return {0: ("비견", "겁재"), 1: ("식신", "상관"), 2: ("편재", "정재"),
            3: ("편관", "정관"), 4: ("편인", "정인")}[d][0 if same else 1]


def conds(yp, mp, dp):
    il = dp[0]
    gs = {god(il, BONGI[mp[1]]), god(il, BONGI[dp[1]])}
    br = {yp[1], mp[1], dp[1]}
    dh = (DOHWA[yp[1]] in br) or (DOHWA[dp[1]] in br)
    return {"상관": "상관" in gs, "식신·상관": bool(gs & {"식신", "상관"}), "편인": "편인" in gs, "도화살": dh}


p = pd.read_csv(PEOPLE)
f = pd.read_csv(FAME)
p = p.merge(f, on="p", how="left")
p["sl"] = p["sl"].fillna(0)
p.loc[p["sl"] < 3, "sl"] = 0
p["occ_list"] = p.occs.str.split("|")
a = p[p.occ_list.apply(lambda o: any(x in ACTOR for x in o))].copy()
C = a.apply(lambda r: conds(r.year_pillar, r.month_pillar, r.day_pillar), axis=1, result_type="expand")
a = pd.concat([a, C], axis=1)
thr = a["sl"].quantile(0.9)
a["top"] = a["sl"] >= thr
L = [f"# 배우 계열 내 사주 조건 vs 유명도 (N={len(a)}, sl≥3인 사람 {(a.sl>=3).sum()}명)", "",
     f"상위 10% 기준 sl ≥ {thr:.0f} (동점 포함 {a.top.sum()}명)", "",
     "| 구분 | 조건 | 보유 N | 비보유 N | 보유 sl 중앙값 | 비보유 중앙값 | 보유 평균 | 비보유 평균 | CLES | U 양측 p | 상위10% 보유 % | 비보유 % | 기준 | 판정 |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for kind, name, alpha in (("1차", "상관", 0.05), ("2차", "식신·상관", 0.05 / 3),
                          ("2차", "편인", 0.05 / 3), ("2차", "도화살", 0.05 / 3)):
    h, n = a[a[name]], a[~a[name]]
    u, pv = stats.mannwhitneyu(h.sl, n.sl, alternative="two-sided")
    cles = u / (len(h) * len(n))
    L.append(f"| {kind} | {name} | {len(h)} | {len(n)} | {h.sl.median():.0f} | {n.sl.median():.0f} | "
             f"{h.sl.mean():.2f} | {n.sl.mean():.2f} | {cles:.3f} | {pv:.3f} | {100*h.top.mean():.1f} | "
             f"{100*n.top.mean():.1f} | {alpha:.4f} | {'통과' if pv < alpha else '통과 못 함'} |")
L += ["", "CLES = 보유자가 비보유자보다 sl이 클 확률(동점 반반). 0.5이면 차이 없음.",
      "참고(생년 교란 점검): 배우 안에서 출생 연도와 sl의 스피어만 상관 = %.3f" % stats.spearmanr(a.year, a.sl)[0]]
open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
