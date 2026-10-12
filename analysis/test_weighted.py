"""월별 출생 가중 재계산 R. 정의는 task10_progress.md 2026-10-12 13:30에 계산 전 고정.
usage: python3 -I analysis/test_weighted.py data/people_main.csv data/fame_raw.csv data/looks_group_raw.txt data/kosis_raw.csv OUT.md
"""
import csv, io, os, sys, calendar, collections
from datetime import date
import numpy as np
import pandas as pd
import sxtwl
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saju_kst import YG, MG, is_term_day

PEOPLE, FAME, LOOKS, KOSIS, OUT = sys.argv[1:6]
GAN = list("갑을병정무기경신임계")
JI = list("자축인묘진사오미신유술해")
ELEM = {g: i // 2 for i, g in enumerate(GAN)}
YANG = {g: (i % 2 == 0) for i, g in enumerate(GAN)}
BONGI = dict(zip(JI, "계기갑을무병정기경신무임"))
HONGYEOM = {"갑": "오", "을": "오", "병": "인", "정": "미", "무": "진", "기": "진", "경": "술", "신": "유", "임": "자", "계": "신"}
DOHWA = {}
for grp, tgt in (("인오술", "묘"), ("사유축", "오"), ("신자진", "유"), ("해묘미", "자")):
    for b in grp:
        DOHWA[b] = tgt
ILJU = [GAN[i % 10] + JI[i % 12] for i in range(60)]
ILIDX = {n: i for i, n in enumerate(ILJU)}
COMEDY_HOST = {"코미디언", "owarai tarento", "TV 사회자", "game show host", "쇼핑 호스트"}
ACTOR = {"배우", "영화 배우", "텔레비전 배우", "연극 배우", "뮤지컬 배우", "성우", "일본의 성우", "스턴트 배우",
         "포르노 배우", "성인용 비디오 배우", "video game actor"}
SINGER = {"가수", "팝 가수", "싱어송라이터", "래퍼", "오페라 가수", "vocalist", "성악가", "bass singer",
          "chanteuse", "child singer", "jazz singer"}


def god(il, o):
    d = (ELEM[o] - ELEM[il]) % 5
    same = YANG[o] == YANG[il]
    return {0: ("비견", "겁재"), 1: ("식신", "상관"), 2: ("편재", "정재"), 3: ("편관", "정관"), 4: ("편인", "정인")}[d][0 if same else 1]


def conds(yp, mp, dp):
    il = dp[0]
    gs = {god(il, BONGI[mp[1]]), god(il, BONGI[dp[1]])}
    br = {yp[1], mp[1], dp[1]}
    a1 = bool(gs & {"식신", "상관"})
    b = (DOHWA[yp[1]] in br) or (DOHWA[dp[1]] in br)
    tgt = (ELEM[il] + 2) % 5
    chars = [yp[0], mp[0], BONGI[yp[1]], BONGI[mp[1]], BONGI[dp[1]]]
    r1 = any(ELEM[c] == tgt for c in chars)
    r2 = HONGYEOM[il] in {yp[1], mp[1], dp[1]}
    return [a1, b, a1 and b, r1, r2]


NAMES = ["A1 식상", "B 도화살", "C 둘 다", "R1 재성", "R2 홍염살"]

# monthly shares
rows = [x for x in csv.reader(io.StringIO(open(KOSIS, encoding="cp949").read())) if len(x) > 4]
h = rows[0]
r = [x for x in rows if x[0] == "전국" and x[1] == "출생아수(명)"][0]
births = collections.defaultdict(dict)
for i in range(4, len(h)):
    if "월" not in h[i] or "." not in h[i]:
        continue
    y, m = h[i].split()[0].split(".")
    if r[i] not in ("", "-"):
        births[int(y)][int(m)] = int(r[i].replace(",", ""))
share = {y: np.array([births[y][m] for m in range(1, 13)], float) for y in births if len(births[y]) == 12}
share = {y: v / v.sum() for y, v in share.items()}
pool = np.mean([share[y] for y in range(1981, 1991)], axis=0)

cache = {}


def year_tab(y):
    if y in cache:
        return cache[y]
    s = share.get(y, pool)
    n = 366 if calendar.isleap(y) else 365
    base = date(y, 1, 1).toordinal()
    W, C, I = [], [], []
    for k in range(n):
        dt = date.fromordinal(base + k)
        if is_term_day(dt):
            continue
        d = sxtwl.fromSolar(dt.year, dt.month, dt.day)
        yg, mg, dg = YG(dt), MG(dt), d.getDayGZ()
        yp, mp, dp = GAN[yg.tg] + JI[yg.dz], GAN[mg.tg] + JI[mg.dz], GAN[dg.tg] + JI[dg.dz]
        W.append(s[dt.month - 1] / calendar.monthrange(y, dt.month)[1])
        C.append(conds(yp, mp, dp))
        I.append(ILIDX[dp])
    W = np.array(W); W /= W.sum()
    U = np.full(len(W), 1.0 / len(W))
    C = np.array(C, float)
    I = np.array(I)
    ilw = np.bincount(I, weights=W, minlength=60)
    cache[y] = (W, U, C, I, ilw)
    return cache[y]


p = pd.read_csv(PEOPLE).merge(pd.read_csv(FAME), on="p", how="left")
p["sl"] = p["sl"].fillna(0)
p["occ"] = p.occs.str.split("|")
p["qid"] = p["p"].str.rsplit("/", n=1).str[-1]
looks = set(t.split(":")[0] for t in open(LOOKS, encoding="utf-8").read().strip().split(","))
F = np.array([conds(r.year_pillar, r.month_pillar, r.day_pillar) for r in p.itertuples()], float)
PW = np.array([year_tab(int(y))[0] @ year_tab(int(y))[2] for y in p.year])
PU = np.array([year_tab(int(y))[1] @ year_tab(int(y))[2] for y in p.year])
thr = float(np.percentile(p.sl, 90))
masks = {
    "전체": np.ones(len(p), bool),
    "P1 코미디언·사회자": p.occ.apply(lambda o: any(x in COMEDY_HOST for x in o)).values,
    "P2 배우 계열": p.occ.apply(lambda o: any(x in ACTOR for x in o)).values,
    "P3 가수 계열": p.occ.apply(lambda o: any(x in SINGER for x in o)).values,
    "P6 배우·모델/미스코리아": p.qid.isin(looks).values,
}
L = ["# 월별 출생 가중 재계산 (KOSIS 1981~, 이전 연도는 1981~1990 평균 비중)", "",
     "| 검정 | N | 관측 % | 균등 기대 % | 가중 기대 % | z(균등) | p(균등) | z(가중) | p(가중) | 기준 0.0042 |", "|---|---|---|---|---|---|---|---|---|---|"]


def row(label, ix, j):
    n = int(ix.sum())
    obs = F[ix, j].sum()
    out = []
    for P in (PU, PW):
        e = P[ix, j].sum(); v = (P[ix, j] * (1 - P[ix, j])).sum()
        z = (obs - e) / v ** 0.5
        out.append((100 * e / n, z, 2 * stats.norm.sf(abs(z))))
    return f"| {label} | {n:,} | {100*obs/n:.1f} | {out[0][0]:.1f} | {out[1][0]:.1f} | {out[0][1]:.2f} | {out[0][2]:.3f} | {out[1][1]:.2f} | {out[1][2]:.3f} | {'통과' if out[1][2] < 0.05/12 else '통과 못 함'} |"


allix = masks["전체"]
for j, nm in enumerate(NAMES):
    L.append(row(nm, allix, j))
L.append(row("P1 코미디언·사회자 식상(A1)", masks["P1 코미디언·사회자"], 0))
L.append(row("P2 배우 계열 도화살(B)", masks["P2 배우 계열"], 1))
L.append(row("P3 가수 계열 도화살(B)", masks["P3 가수 계열"], 1))
L.append(row("P6 배우·모델/미스코리아 도화살(B)", masks["P6 배우·모델/미스코리아"], 1))
# P4
top = (p.sl >= thr).values
res = []
for P in (PU, PW):
    d = []
    for ix in (top, ~top):
        n = ix.sum(); o = F[ix, 2].mean(); e = P[ix, 2].mean(); v = (P[ix, 2] * (1 - P[ix, 2])).sum() / n ** 2
        d.append((o - e, v))
    z = (d[0][0] - d[1][0]) / (d[0][1] + d[1][1]) ** 0.5
    res.append((100 * (d[0][0] - d[1][0]), z, 2 * stats.norm.sf(abs(z))))
L.append(f"| P4 유명도 상위(sl≥{thr:.0f}) C 초과분 − 나머지 | {int(top.sum()):,} / {int((~top).sum()):,} | | | | {res[0][1]:.2f} ({res[0][0]:+.1f}%p) | {res[0][2]:.3f} | {res[1][1]:.2f} ({res[1][0]:+.1f}%p) | {res[1][2]:.3f} | {'통과' if res[1][2] < 0.05/12 else '통과 못 함'} |")
# 일주
obs = np.bincount([ILIDX[d] for d in p.day_pillar], minlength=60).astype(float)
exp_u = np.zeros(60); exp_w = np.zeros(60)
for y in p.year:
    W, U, C, I, ilw = year_tab(int(y))
    exp_u += np.bincount(I, weights=U, minlength=60); exp_w += ilw
chi_u = ((obs - exp_u) ** 2 / exp_u).sum(); chi_w = ((obs - exp_w) ** 2 / exp_w).sum()
N = len(p)
L += ["", f"일주 분포(59 df): 균등 기대 χ²={chi_u:.1f}, p={stats.chi2.sf(chi_u,59):.3f}, w={np.sqrt(chi_u/N):.3f} / 가중 기대 χ²={chi_w:.1f}, p={stats.chi2.sf(chi_w,59):.3f}, w={np.sqrt(chi_w/N):.3f}",
      "", "월 비중(1981~1990 평균, %): " + ", ".join(f"{100*x:.1f}" for x in pool)]
open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
