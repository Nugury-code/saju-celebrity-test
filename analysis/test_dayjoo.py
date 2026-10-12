"""Step 2: is the 일주 (day pillar) distribution of Korean entertainers different
from what the calendar alone would produce?

Null model (fixed in advance): a person born in year Y is equally likely to be
born on any day of Y, so the expected 일주 distribution for that person is the
share of each 일주 over the days of year Y. Expected counts = sum over people.
Test: Pearson chi-square, 59 df, plus a Monte Carlo p-value from the same null.
Effect size: w = sqrt(chi2 / N).
"""
import sys
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saju_kst import is_term_day
DROP = os.environ.get('SAJU_DROP_TERM_DAYS') == '1'  # main analysis: leave out 절-day births and the same days of the baseline calendar

SRC, OUT_MD, OUT_CSV = sys.argv[1], sys.argv[2], sys.argv[3]
GAN = list("갑을병정무기경신임계")
JI = list("자축인묘진사오미신유술해")
NAMES = [GAN[i % 10] + JI[i % 12] for i in range(60)]
rng = np.random.default_rng(20261008)  # fixed seed for reproducibility

ACTOR = {"배우", "영화 배우", "텔레비전 배우", "연극 배우", "뮤지컬 배우", "성우",
         "일본의 성우", "스턴트 배우", "포르노 배우", "성인용 비디오 배우",
         "video game actor"}
SINGER = {"가수", "팝 가수", "싱어송라이터", "래퍼", "오페라 가수", "vocalist", "성악가",
          "bass singer", "chanteuse", "child singer", "jazz singer"}


def day_idx(y, m, d):
    return (date(y, m, d).toordinal() + 1721425 - 11) % 60


_year_cache = {}


def year_dist(y):
    if y not in _year_cache:
        n = 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
        base = date(y, 1, 1).toordinal()
        cnt = np.zeros(60)
        for k in range(n):
            dt = date.fromordinal(base + k)
            if DROP and is_term_day(dt):
                continue
            cnt[day_idx(dt.year, dt.month, dt.day)] += 1
        _year_cache[y] = cnt / cnt.sum()
    return _year_cache[y]


def run(sub, label, sims=20000):
    n = len(sub)
    obs = np.bincount(sub.day_idx.to_numpy(), minlength=60).astype(float)
    exp = np.zeros(60)
    for y, c in sub.year.value_counts().items():
        exp += c * year_dist(int(y))
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    p_asym = float(stats.chi2.sf(chi2, 59))
    probs = exp / exp.sum()
    sim = rng.multinomial(n, probs, size=sims)
    sim_chi = ((sim - exp) ** 2 / exp).sum(axis=1)
    p_mc = float((np.sum(sim_chi >= chi2) + 1) / (sims + 1))
    w = (chi2 / n) ** 0.5
    z = (obs - exp) / np.sqrt(exp)
    return dict(label=label, n=n, chi2=chi2, p_asym=p_asym, p_mc=p_mc, w=w,
                exp_min=float(exp.min()), obs=obs, exp=exp, z=z,
                sim_w_95=float((np.percentile(sim_chi, 95) / n) ** 0.5))


p = pd.read_csv(SRC)
p["occ_list"] = p.occs.str.split("|")
p["is_actor"] = p.occ_list.apply(lambda o: any(x in ACTOR for x in o))
p["is_singer"] = p.occ_list.apply(lambda o: any(x in SINGER for x in o))
p["day_idx"] = p.day_pillar.map({n: i for i, n in enumerate(NAMES)})

specs = [
    ("주 분석: 전체(포르노 배우 포함, 43종 전부)", p),
    ("민감도1: 포르노 배우 범주 제외", p[~p.is_porn]),
    ("민감도2: 애매한 직업만 가진 사람 제외", p[~p.only_ambiguous]),
    ("민감도3: 1월 1일 생 제외(기본값 의심)", p[~p.jan1]),
    ("민감도4: 생일이 2개 이상 등록된 사람 제외", p[~p.multi_dob]),
    ("민감도5: 위 1~4 모두 제외", p[~p.is_porn & ~p.only_ambiguous & ~p.jan1 & ~p.multi_dob]),
    ("보조: 배우 계열만", p[p.is_actor]),
    ("보조: 가수 계열만", p[p.is_singer]),
]
res = [run(s, lab) for lab, s in specs]

lines = ["| 분석 | N | 카이제곱(df=59) | p(점근) | p(모의 2만회) | 효과 크기 w | 우연만으로도 나올 w 상한(95%) |",
         "|---|---|---|---|---|---|---|"]
for r in res:
    lines.append(f"| {r['label']} | {r['n']:,} | {r['chi2']:.1f} | {r['p_asym']:.3f} | {r['p_mc']:.3f} | {r['w']:.3f} | {r['sim_w_95']:.3f} |")
open(OUT_MD, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))

main = res[0]
cells = pd.DataFrame({"일주": NAMES, "관측": main["obs"].astype(int),
                      "기대": main["exp"].round(1), "표준화잔차z": main["z"].round(2)})
cells["|z|"] = cells["표준화잔차z"].abs()
cells.sort_values("|z|", ascending=False).drop(columns="|z|").to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
top = cells.sort_values("|z|", ascending=False).head(8)
print("\n주 분석 칸별 상위 8 (60칸 중 |z|>2는 우연으로도 약 3칸, Bonferroni 기준 |z|>3.34):")
print(top.drop(columns="|z|").to_string(index=False))
print("주 분석 |z|>2 칸 수:", int((cells['|z|'] > 2).sum()), "| |z|>3.34 칸 수:", int((cells['|z|'] > 3.34).sum()))
print("기대 인원 최소 칸:", round(main["exp_min"], 1))
