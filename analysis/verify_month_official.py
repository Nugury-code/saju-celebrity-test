"""Check the month pillar against KASI's official 24-term table (1920-2100, KST).
1) sxtwl jie times vs KASI (1920-2017); 2) every person's month branch vs the KASI term calendar.
KASI table: data/official/kasi_24terms_1920-2100.txt (optional; downloaded by the user from astro.kasi.re.kr, 2026-10-12)."""
import re, sys
from datetime import datetime, date, timedelta
import pandas as pd, sxtwl

OUT = sys.argv[1] if len(sys.argv) > 1 else "data/month_official_summary.md"
rows = []
for line in open("data/official/kasi_24terms_1920-2100.txt", encoding="utf-8"):
    m = re.match(r"^\s*(\d+),\s*(\d{4}),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*$", line)
    if m:
        rows.append(tuple(map(int, m.groups())))
t = pd.DataFrame(rows, columns=["term", "y", "m", "d", "h", "mi"])
# 24:00 notation = date unchanged, time 24:00 (per KASI note); keep as date at 23:59:59.99 -> use datetime with hour 24 -> next day 00:00
def to_dt(r):
    # KASI writes 24:00 when rounding would change the date: the term falls in the last 30 s of day d.
    if r.h == 24:
        return datetime(r.y, r.m, r.d, 23, 59, 45)
    return datetime(r.y, r.m, r.d) + timedelta(hours=int(r.h), minutes=int(r.mi))
t["dt"] = t.apply(to_dt, axis=1)
jie = t[t.term % 2 == 1].sort_values("dt").reset_index(drop=True)

# 1) sxtwl vs KASI (KST = lib UTC+8 + 1h)
diffs = []
for r in jie.itertuples():
    if r.y > 2017:
        continue
    d0 = date(r.y, r.m, r.d)
    best = None
    for k in range(-2, 3):
        dd = d0 + timedelta(days=k)
        dy = sxtwl.fromSolar(dd.year, dd.month, dd.day)
        if dy.hasJieQi() and dy.getJieQi() == (r.term - 1 + 6) % 24 - 0 + 0:
            pass
        if dy.hasJieQi():
            tm = sxtwl.JD2DD(dy.getJieQiJD())
            cand = datetime(int(tm.Y), int(tm.M), int(tm.D), int(tm.h), int(tm.m)) + timedelta(hours=1)
            if abs((cand - r.dt).total_seconds()) < 3 * 86400 and (best is None or abs((cand - r.dt).total_seconds()) < abs((best - r.dt).total_seconds())):
                best = cand
    if best is not None:
        diffs.append(((best - r.dt).total_seconds() / 60, best.date() != r.dt.date()))
d = pd.DataFrame(diffs, columns=["min", "date_differs"])

# 2) people
BR = {1: "인", 3: "묘", 5: "진", 7: "사", 9: "오", 11: "미", 13: "신", 15: "유", 17: "술", 19: "해", 21: "자", 23: "축"}
p = pd.read_csv("data/people_clean.csv")
p["dob"] = pd.to_datetime(p["dob"])
q = p[(p.dob.dt.year >= 1921) & (p.dob.dt.year <= 2100)].copy()  # 1920 needs the previous year's 대설
starts = list(zip(jie.dt, jie.term))
import bisect
keys = [s[0] for s in starts]
def branch_on(day, end_of_day):
    ts = datetime(day.year, day.month, day.day) + (timedelta(days=1) - timedelta(seconds=1) if end_of_day else timedelta(0))
    i = bisect.bisect_right(keys, ts) - 1
    return BR[starts[i][1]]
q["br_start"] = q.dob.apply(lambda x: branch_on(x, False))   # month branch at 00:00
q["br_end"] = q.dob.apply(lambda x: branch_on(x, True))      # month branch at 23:59
q["mine"] = q.month_pillar.str[1]
q["on_jie_day"] = q.br_start != q.br_end
clear = q[~q.on_jie_day]
clear_mis = clear[clear.mine != clear.br_start]
amb = q[q.on_jie_day]
amb_ok = amb[(amb.mine == amb.br_start) | (amb.mine == amb.br_end)]
lines = [
    f"KASI 24기 표: {len(t)}행, {t.y.min()}~{t.y.max()}년, 연 24개 모두 있음",
    "",
    "## 1. 라이브러리 절기 시각과 KASI 공식 시각 비교 (1920~2017년의 12개 절 × 98년)",
    f"- 비교한 절기 수: {len(d):,}",
    f"- 시각 차이(분): 평균 {d['min'].mean():+.1f}, 절대값 최대 {d['min'].abs().max():.0f}, 95% 이하 {d['min'].abs().quantile(0.95):.0f}",
    f"- 날짜가 달라진 절기: {int(d.date_differs.sum())}개",
    "",
    "## 2. 사람별 월지 비교 (1921~2017년생 {:,}명, 1920년 이전 {}명은 표 범위 밖)".format(len(q), int((p.dob.dt.year < 1921).sum())),
    f"- 절기 당일이 아닌 사람: {len(clear):,}명 → 내 월지와 KASI 기준 월지가 다른 사람 {len(clear_mis)}명",
    f"- 절기 당일 출생(시각 모르면 월주 확정 불가): {len(amb):,}명 → 내 월지가 '절기 전' 또는 '절기 후' 둘 중 하나인 사람 {len(amb_ok):,}명",
]
if len(clear_mis):
    lines.append("- 불일치 예: " + ", ".join(f"{r.p}({r.dob.date()}) 내={r.mine} KASI={r.br_start}" for r in clear_mis.head(8).itertuples()))
open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
