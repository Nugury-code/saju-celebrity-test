"""Independent check of the solar-term (jie) dates used for the month pillar.
Sun's apparent longitude by Meeus' low-precision formula (ch. 25, ~0.01 deg), solved by bisection.
Compared with the sxtwl library's jie time (library time = UTC+8). Reads no project data."""
import math, sxtwl
from datetime import datetime, timedelta, timezone

def jd_from_dt(dt):  # dt UTC
    y,m = dt.year, dt.month
    d = dt.day + (dt.hour+dt.minute/60+dt.second/3600)/24
    if m<=2: y-=1; m+=12
    A=y//100; B=2-A+A//4
    return int(365.25*(y+4716))+int(30.6001*(m+1))+d+B-1524.5

def sun_lon(jd):
    T=(jd-2451545.0)/36525
    L0=280.46646+36000.76983*T+0.0003032*T*T
    M=math.radians(357.52911+35999.05029*T-0.0001537*T*T)
    C=(1.914602-0.004817*T-0.000014*T*T)*math.sin(M)+(0.019993-0.000101*T)*math.sin(2*M)+0.000289*math.sin(3*M)
    om=math.radians(125.04-1934.136*T)
    return (L0+C-0.00569-0.00478*math.sin(om))%360

def crossing(year, target, guess_month, guess_day):
    lo=datetime(year,guess_month,guess_day,tzinfo=timezone.utc)-timedelta(days=4)
    hi=lo+timedelta(days=8)
    f=lambda dt:((sun_lon(jd_from_dt(dt))-target+180)%360)-180
    for _ in range(60):
        mid=lo+(hi-lo)/2
        if f(mid)>0: hi=mid
        else: lo=mid
    return lo+(hi-lo)/2

cases=[("백로",1903,9,9,165),("청명",1970,4,5,15),("청명",1978,4,5,15),("입동",1974,11,8,225),("소서",1987,7,7,105)]
print("절기 | 연 | 독립계산(UTC) | 독립계산(UTC+8) | sxtwl(라이브러리 시각) | 차이(분)")
for name,y,m,d,lon in cases:
    t=crossing(y,lon,m,d)
    t8=t+timedelta(hours=8)
    day=sxtwl.fromSolar(y,m,d)
    # find sxtwl jie within +-3 days
    from datetime import date
    best=None
    for k in range(-3,4):
        dd=date(y,m,d)+timedelta(days=k)
        dy=sxtwl.fromSolar(dd.year,dd.month,dd.day)
        if dy.hasJieQi():
            tm=sxtwl.JD2DD(dy.getJieQiJD())
            best=datetime(int(tm.Y),int(tm.M),int(tm.D),int(tm.h),int(tm.m))
    diff=(t8.replace(tzinfo=None)-best).total_seconds()/60
    print(f"{name} | {y} | {t:%Y-%m-%d %H:%M} | {t8:%Y-%m-%d %H:%M} | {best:%Y-%m-%d %H:%M} | {diff:+.0f}")
