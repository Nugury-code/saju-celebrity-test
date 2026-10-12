"""카드 4 점검 K1~K3. 정의는 task10_progress.md 2026-10-12 11:55에 계산 전 고정.
usage: python3 -I analysis/card4_checks.py data/people_main.csv data/fame_raw.csv OUT.md
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
ACTOR = {"배우", "영화 배우", "텔레비전 배우", "연극 배우", "뮤지컬 배우", "성우",
         "일본의 성우", "스턴트 배우", "포르노 배우", "성인용 비디오 배우", "video game actor"}


def god(il, o):
    d = (ELEM[o] - ELEM[il]) % 5
    same = YANG[o] == YANG[il]
    return {0: ("비견", "겁재"), 1: ("식신", "상관"), 2: ("편재", "정재"),
            3: ("편관", "정관"), 4: ("편인", "정인")}[d][0 if same else 1]


p = pd.read_csv(PEOPLE).merge(pd.read_csv(FAME), on="p", how="left")
p["sl_raw"] = p["sl"].fillna(0)
p["sl3"] = p["sl_raw"].where(p["sl_raw"] >= 3, 0)
p["occ_list"] = p.occs.str.split("|")
a = p[p.occ_list.apply(lambda o: any(x in ACTOR for x in o))].copy()
a["sg"] = a.apply(lambda r: "상관" in {god(r.day_pillar[0], BONGI[r.month_pillar[1]]),
                                       god(r.day_pillar[0], BONGI[r.day_pillar[1]])}, axis=1)
L = ["# 카드 4 점검 (배우 계열 N=%d, 상관 보유 %d)" % (len(a), a.sg.sum()), ""]

# K1
a["dec"] = a.year.astype(int) // 10 * 10
rows, zs, ws = [], [], []
for d, g in a.groupby("dec"):
    h, n = g[g.sg], g[~g.sg]
    if len(h) < 5 or len(n) < 5:
        rows.append(f"| {d}년대 | {len(h)} | {len(n)} | 제외(5명 미만) | |")
        continue
    u, _ = stats.mannwhitneyu(h.sl3, n.sl3, alternative="two-sided")
    n1, n2 = len(h), len(n)
    allv = np.concatenate([h.sl3.values, n.sl3.values])
    N = n1 + n2
    _, t = np.unique(allv, return_counts=True)
    var = n1 * n2 / 12 * ((N + 1) - (t ** 3 - t).sum() / (N * (N - 1)))
    z = (u - n1 * n2 / 2) / np.sqrt(var) if var > 0 else 0.0
    zs.append(z); ws.append(np.sqrt(N))
    rows.append(f"| {d}년대 | {n1} | {n2} | {z:.2f} | {u/(n1*n2):.3f} |")
zc = np.sum(np.array(zs) * np.array(ws)) / np.sqrt(np.sum(np.array(ws) ** 2))
L += ["## K1 출생 연대별 층화", "", "| 연대 | 보유 N | 비보유 N | z | CLES |", "|---|---|---|---|---|"] + rows
L += ["", f"가중 합산 z = {zc:.2f}, 양측 p = {2*stats.norm.sf(abs(zc)):.3f}", ""]

# K2
h, n = a[a.sg], a[~a.sg]
u, pv = stats.mannwhitneyu(h.sl_raw, n.sl_raw, alternative="two-sided")
L += ["## K2 유명도를 있는 값 그대로 사용(sl<3도 유지, 없으면 0)", "",
      f"보유 평균 {h.sl_raw.mean():.2f} vs 비보유 {n.sl_raw.mean():.2f}, CLES {u/(len(h)*len(n)):.3f}, 양측 p = {pv:.3f}", ""]

# K3
n1, n2 = int(a.sg.sum()), int((~a.sg).sum())
N = n1 + n2
_, t = np.unique(a.sl3.values, return_counts=True)
var = n1 * n2 / 12 * ((N + 1) - (t ** 3 - t).sum() / (N * (N - 1)))
se = np.sqrt(var) / (n1 * n2)  # SE of CLES under H0 (tie corrected)
mde = (stats.norm.isf(0.025) + stats.norm.ppf(0.8)) * se
L += ["## K3 검출력", "", f"H0에서 CLES의 표준오차(동점 보정) = {se:.4f}",
      f"α=0.05 양측·검출력 80%에서 잡을 수 있는 최소 CLES ≈ {0.5+mde:.3f} (또는 {0.5-mde:.3f})",
      "(동점이 많아 유효 정보량이 줄어든 상태를 반영한 값. 대략적 근사)"]
open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
