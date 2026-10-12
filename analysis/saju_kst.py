"""Year and month pillar with 절(節) dates in Korean time (KST, UTC+9).

Why: the sxtwl library dates its solar terms in Beijing time (UTC+8). A 절 that falls between
00:00 and 00:59 KST is on the previous calendar day in Beijing, so sxtwl starts the new month one
day early for Korean birth dates (found by checking against KASI's official table, 2026-10-12).

Source of the term dates:
  1920-2100: KASI "24기 입기 시각" table (data/official/kasi_24terms_1920-2100.txt), KST.
  before 1920: sxtwl term time + 1 hour (sxtwl matched KASI within 1 minute on all 1,176 절 in 1920-2017).
Convention (unchanged from the sxtwl version): the day a 절 falls on already belongs to the new month/year.
Day pillar is not affected (it depends only on the calendar date).
"""
import os, re
from collections import namedtuple
from datetime import date, datetime, timedelta

import sxtwl

HERE = os.path.dirname(os.path.abspath(__file__))
PIL = namedtuple("PIL", "tg dz")
BRANCH_OF_TERM = {1: 2, 3: 3, 5: 4, 7: 5, 9: 6, 11: 7, 13: 8, 15: 9, 17: 10, 19: 11, 21: 0, 23: 1}  # 입춘->인(2) ... 소한->축(1)

_terms = {}   # date -> KASI term number (odd = 절)


# sxtwl's getJieQi index: 0 = 冬至 ... odd/even meaning differs from KASI's numbering, so for the
# pre-1920 part we derive the KASI-style number from the term's solar longitude order instead.
def _add_sxtwl(y0, y1):
    """Fill terms for years y0..y1-1 from sxtwl (Beijing time) + 1 hour = Korean time."""
    for y in range(y0, y1):
        for k in range(366):
            dt = date(y, 1, 1) + timedelta(days=k)
            if dt.year != y:
                break
            x = sxtwl.fromSolar(dt.year, dt.month, dt.day)
            if x.hasJieQi():
                kasi = (x.getJieQi() + 21) % 24 + 1   # sxtwl 0=冬至(KASI 22), 1=小寒(23) ...
                if kasi % 2 == 1:
                    tm = sxtwl.JD2DD(x.getJieQiJD())
                    kst = datetime(int(tm.Y), int(tm.M), int(tm.D), int(tm.h), int(tm.m)) + timedelta(hours=1)
                    _terms[kst.date()] = kasi


def _build():
    path = os.path.join(HERE, "..", "data", "official", "kasi_24terms_1920-2100.txt")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            m = re.match(r"^\s*(\d+),\s*(\d{4}),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*$", line)
            if m:
                term, y, mo, d, h, mi = map(int, m.groups())
                if term % 2 == 1:
                    _terms[date(y, mo, d)] = term
        _add_sxtwl(1850, 1920)
    else:
        # KASI table not present (it is not redistributed in the public repository):
        # use sxtwl + 1 hour for every year.
        _add_sxtwl(1850, 2021)


_build()
_dates = sorted(_terms)
import bisect


def _latest_jie(dt):
    i = bisect.bisect_right(_dates, dt) - 1
    return _dates[i], _terms[_dates[i]]


def _lichun(year):
    for dd in _dates:
        if dd.year == year and _terms[dd] == 1:
            return dd
    raise ValueError(year)


def year_pillar(dt):
    y = dt.year if dt >= _lichun(dt.year) else dt.year - 1
    idx = (y - 4) % 60
    return PIL(idx % 10, idx % 12)


def month_pillar(dt):
    jdate, term = _latest_jie(dt)
    branch = BRANCH_OF_TERM[term]
    ystem = year_pillar(dt).tg
    # 오호둔: 인월의 천간 = (년간 % 5) * 2 + 2
    first = ((ystem % 5) * 2 + 2) % 10
    offset = (branch - 2) % 12                        # 인=0 ... 축=11
    return PIL((first + offset) % 10, branch)


def YG(dt):  # dt: datetime.date
    return year_pillar(dt)


def MG(dt):
    return month_pillar(dt)


def is_term_day(dt):
    """True if a 절(節) falls on this calendar date in KST (month/year pillar then depends on birth time)."""
    return dt in _terms
