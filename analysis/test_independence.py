"""Step 7 (P5): is 일주 independent of occupation group?

Groups (one per person, priority order, fixed in the progress log before running):
  1 코미디언·사회자  ->  2 가수  ->  3 배우  ;  anyone in none of the three lists is "기타" and excluded.
Statistic: Pearson chi-square on the 60 x 3 table (df = 59*2 = 118) and a permutation p-value
(group labels shuffled across people, 20,000 times, seed 20261008). Effect size: Cramer's V.
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

SRC, OUT_MD = sys.argv[1], sys.argv[2]
GAN = list("갑을병정무기경신임계")
JI = list("자축인묘진사오미신유술해")
NAMES = [GAN[i % 10] + JI[i % 12] for i in range(60)]
COMEDY_HOST = {"코미디언", "owarai tarento", "TV 사회자", "game show host", "쇼핑 호스트"}
SINGER = {"가수", "팝 가수", "싱어송라이터", "래퍼", "오페라 가수", "vocalist", "성악가",
          "bass singer", "chanteuse", "child singer", "jazz singer"}
ACTOR = {"배우", "영화 배우", "텔레비전 배우", "연극 배우", "뮤지컬 배우", "성우",
         "일본의 성우", "스턴트 배우", "포르노 배우", "성인용 비디오 배우", "video game actor"}
LABELS = ["코미디언·사회자", "가수", "배우"]
rng = np.random.default_rng(20261008)


def group_of(occs):
    s = set(occs)
    flags = [bool(s & COMEDY_HOST), bool(s & SINGER), bool(s & ACTOR)]
    first = next((i for i, f in enumerate(flags) if f), None)
    return first, sum(flags)


def chi2_table(day, grp, nday=60, ngrp=3):
    tab = np.bincount(day * ngrp + grp, minlength=nday * ngrp).reshape(nday, ngrp).astype(float)
    row, col = tab.sum(1, keepdims=True), tab.sum(0, keepdims=True)
    exp = row @ col / tab.sum()
    return float(((tab - exp) ** 2 / exp).sum()), tab, exp


def run(df, label, sims=20000):
    day = df.day_idx.to_numpy()
    grp = df.grp.to_numpy()
    n = len(df)
    chi, tab, exp = chi2_table(day, grp)
    dof = 59 * 2
    p_asym = float(stats.chi2.sf(chi, dof))
    sim = np.empty(sims)
    for k in range(sims):
        sim[k] = chi2_table(day, rng.permutation(grp))[0]
    p_perm = float((np.sum(sim >= chi) + 1) / (sims + 1))
    v = (chi / (n * 2)) ** 0.5
    v95 = (np.percentile(sim, 95) / (n * 2)) ** 0.5
    return dict(label=label, n=n, chi=chi, dof=dof, p_asym=p_asym, p_perm=p_perm, v=v, v95=v95,
                counts=[int((grp == g).sum()) for g in range(3)], min_exp=float(exp.min()))


p = pd.read_csv(SRC)
p["day_idx"] = p.day_pillar.map({n: i for i, n in enumerate(NAMES)})
g = p.occs.str.split("|").apply(group_of)
p["grp"] = g.apply(lambda t: t[0])
p["ntypes"] = g.apply(lambda t: t[1])
other = int(p.grp.isna().sum())
q = p[p.grp.notna()].copy()
q["grp"] = q.grp.astype(int)

res = [run(q, "주 분석: 3개 직업군 전체"), run(q[q.ntypes == 1], "민감도: 직업군이 한 종류뿐인 사람만")]
lines = [f"기타(세 직업군 어디에도 없음)로 제외된 인원: {other}명", "",
         "| 분석 | N | 코미디언·사회자 / 가수 / 배우 | 카이제곱 (자유도 118) | p(점근) | p(순열 2만회) | Cramér's V | 우연만으로 나올 V 상한(95%) | 기준(0.0045) |",
         "|---|---|---|---|---|---|---|---|---|"]
for r in res:
    c = " / ".join(f"{x:,}" for x in r["counts"])
    lines.append(f"| {r['label']} | {r['n']:,} | {c} | {r['chi']:.1f} | {r['p_asym']:.3f} | {r['p_perm']:.3f} | "
                 f"{r['v']:.3f} | {r['v95']:.3f} | {'통과' if r['p_perm'] < 0.0045 else '통과 못 함'} |")
lines.append("")
lines.append(f"기대 인원이 가장 작은 칸(주 분석): {res[0]['min_exp']:.1f}명. "
             + ("모든 칸이 5명 이상이라 점근 p값과 순열 p값이 거의 같음. 판정은 순열 p값 기준." if res[0]['min_exp'] >= 5
                else "5명 미만인 칸이 있어 순열 p값을 기준으로 삼음."))
open(OUT_MD, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
