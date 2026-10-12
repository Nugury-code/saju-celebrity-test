"""Step 1: raw Wikidata CSV -> one row per person, with 일주/월주/년주.

Input : data/query_raw.csv  (columns: p, dob, prec, occLabel)
Output: data/people_clean.csv
Day pillar is computed two independent ways (Julian-day formula and the
sxtwl library) and the script stops if they ever disagree.
"""
import sys
from datetime import date

import pandas as pd
import sxtwl
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from saju_kst import YG, MG  # KST 절기 (see saju_kst.py)

RAW, OUT = sys.argv[1], sys.argv[2]

GAN = list("갑을병정무기경신임계")
JI = list("자축인묘진사오미신유술해")
JIE_ODD = set(range(1, 24, 2))  # 절(節) indices in sxtwl: 소한, 입춘, 경칩 ...

# Fixed BEFORE looking at any result (see progress log): occupations that are
# doubtful as "연예인". A person is flagged only if ALL of their occupations
# are in this list. Used for the sensitivity analysis only.
AMBIGUOUS = {
    "뉴스 캐스터", "기상 캐스터", "스포츠캐스터", "스포츠 해설가",
    "esports commentator", "baseball commentator", "soccer commentator",
    "voice teacher", "opera vocal coach", "스턴트 코디네이터",
    "쇼핑 호스트", "코스플레이어", "audiobook narrator", "game show host",
}
PORN = {"포르노 배우", "성인용 비디오 배우"}


def gz(idx):
    return GAN[idx % 10] + JI[idx % 12]


def jdn(y, m, d):
    return date(y, m, d).toordinal() + 1721425  # proleptic Gregorian JDN


def day_idx_formula(y, m, d):
    # 1900-01-01 is 갑술(10); verified: 2000-01-01 is 무오(54)
    return (jdn(y, m, d) - 11) % 60


df = pd.read_csv(RAW)
df["d"] = pd.to_datetime(df["dob"].str[:10], errors="coerce")
print("rows:", len(df), "| unparsable dates:", int(df.d.isna().sum()))

# one row per person; collect occupations
occ = df.groupby("p")["occLabel"].apply(lambda s: sorted(set(s)))
dobs = df.groupby("p")["d"].nunique()
print("people with >1 distinct birth date:", int((dobs > 1).sum()))
people = (df.sort_values("d").drop_duplicates("p")[["p", "d"]]
          .set_index("p"))
people["occs"] = occ
people["n_occ"] = people.occs.apply(len)
people["is_porn"] = people.occs.apply(lambda o: any(x in PORN for x in o))
people["only_ambiguous"] = people.occs.apply(lambda o: all(x in AMBIGUOUS for x in o))
people["multi_dob"] = dobs.reindex(people.index) > 1  # conflicting dates in Wikidata; earliest kept
people = people.reset_index()

rows = []
mismatch = 0
for r in people.itertuples():
    y, m, d = r.d.year, r.d.month, r.d.day
    day = sxtwl.fromSolar(y, m, d)
    gzobj = day.getDayGZ()
    idx_lib = next(i for i in range(60) if i % 10 == gzobj.tg and i % 12 == gzobj.dz)
    idx_f = day_idx_formula(y, m, d)
    if idx_lib != idx_f:
        mismatch += 1
    mg = MG(r.d.date())   # 절 dates in KST (KASI table); sxtwl's own are Beijing time
    yg = YG(r.d.date())   # 입춘 기준, KST
    # month pillar is uncertain on/next to a 절 day (birth time unknown, and
    # sxtwl computes terms in Beijing time, Korea is one hour ahead)
    near_jie = False
    for off in (-1, 0, 1):
        dd = date.fromordinal(r.d.toordinal() + off)
        x = sxtwl.fromSolar(dd.year, dd.month, dd.day)
        if x.hasJieQi() and x.getJieQi() in JIE_ODD:
            near_jie = True
    rows.append({
        "day_idx": idx_f,
        "day_pillar": gz(idx_f),
        "month_branch": JI[mg.dz],
        "month_pillar": GAN[mg.tg] + JI[mg.dz],
        "year_pillar": GAN[yg.tg] + JI[yg.dz],
        "near_jie_day": near_jie,
    })

people = pd.concat([people, pd.DataFrame(rows)], axis=1)
people["jan1"] = (people.d.dt.month == 1) & (people.d.dt.day == 1)
people["year"] = people.d.dt.year
people["occs"] = people.occs.apply(lambda o: "|".join(o))
people["dob"] = people.d.dt.strftime("%Y-%m-%d")
people = people.drop(columns=["d"])
people.to_csv(OUT, index=False)

print("day-pillar formula vs sxtwl mismatches:", mismatch)
if mismatch:
    sys.exit("day pillar: the two methods disagree, stopping")
print("people written:", len(people))
print("jan1 flagged:", int(people.jan1.sum()),
      "| near-절 flagged:", int(people.near_jie_day.sum()),
      "| porn:", int(people.is_porn.sum()),
      "| only-ambiguous:", int(people.only_ambiguous.sum()))
print(people[["dob", "year_pillar", "month_pillar", "day_pillar"]].head(8).to_string())
