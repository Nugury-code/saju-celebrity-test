# 사주 조건은 연예인의 직업과 성취와 관련이 있는가

한국 연예인 11,829명(위키데이터)의 출생 연·월·일로 사주 조건을 계산해, 같은 출생 연도의 달력 날짜만으로 기대되는 비율과 비교한 연구입니다. 논문 전문은 [`paper/paper.pdf`](paper/paper.pdf)입니다.

- **질문:** 연예인은 특정 사주 조건(일주, 식상, 도화살 등)을 달력 기대보다 더 많이 가지는가? 배우 중 상관 보유자의 유명도가 더 높은가?
- **방법:** 절기(KASI 24기 입기 시각, 한국시각)로 연·월주, 율리우스일로 일주를 계산하고 달력 기대 분포와 검정했다. 12개 검정에 다중검정 보정(α=0.05/12)을 적용했다.
- **결과:** 보정 기준을 통과한 검정은 없었다. 차이는 대부분 1%p 미만이었고, 배우 안 상관과 유명도도 차이가 없었다(CLES 0.508, p=0.330). 이 표본으로 약 1%p 이상의 차이는 잡을 수 있었으므로, 검증한 조건에서는 영향이 없다고 볼 수 있다. 시주와 격국·용신 등은 검증하지 않았다.

## 개인정보 처리
위키데이터의 연예인 이름과 출생일 등 **인물별 자료는 이 저장소에 넣지 않았습니다.** 집계 결과만 공개합니다. 이유와 입력 형식은 [`data/README.md`](data/README.md)에 적었습니다.

## 결과 보기
논문 전문은 [`paper/paper.pdf`](paper/paper.pdf)이고, 논문의 표는 `data/main/`의 파일과 같은 값입니다.

| 논문 | 결과 파일 (`data/main/`) |
|---|---|
| 표 1 (분석 1) | `dayjoo_summary.md`, `pattern_summary.md`, `extra_summary.md`, `predictions_summary.md`, `fame_summary.md`(P4), `independence_summary.md`(P5), `looks_summary.md`(P6) |
| 표 2 (분석 2) | `aptitude_fame_summary.md` |
| 3.3 점검 | `card4_checks.md`, `weighted_summary.md` |

`data/full_sample/`에는 제외 기준을 적용하기 전 전체 12,265명의 결과가 있습니다. 난수는 시드 20261008로 고정했습니다.

## 코드 실행
Python 3과 `pip install pandas numpy scipy sxtwl`이 필요합니다. 인물별 입력 파일을 직접 만들어야 하고(형식은 `data/README.md`), 있으면 이 폴더에서 아래 순서로 실행합니다. KASI 24기 표가 `data/official/`에 없으면 sxtwl의 절기 시각에 1시간을 더한 한국시각으로 대신 계산합니다.
```
python3 analysis/build.py data/query_raw.csv data/people_clean.csv
python3 analysis/make_main.py
export SAJU_DROP_TERM_DAYS=1
python3 analysis/test_dayjoo.py data/people_main.csv data/main/dayjoo_summary.md data/main/dayjoo_cells.csv
python3 analysis/test_pattern.py data/people_main.csv data/main/pattern_summary.md
python3 analysis/test_extra.py data/people_main.csv data/main/extra_summary.md
python3 analysis/test_predictions.py data/people_main.csv data/main/predictions_summary.md
python3 analysis/test_fame.py data/people_main.csv data/fame_raw.csv data/main/fame_summary.md
python3 analysis/test_independence.py data/people_main.csv data/main/independence_summary.md
python3 analysis/test_looks.py data/people_main.csv data/looks_group_raw.txt data/main/looks_summary.md
python3 analysis/test_aptitude_fame.py data/people_main.csv data/fame_raw.csv data/main/aptitude_fame_summary.md
python3 analysis/card4_checks.py data/people_main.csv data/fame_raw.csv data/main/card4_checks.md
python3 analysis/test_weighted.py data/people_main.csv data/fame_raw.csv data/looks_group_raw.txt data/kosis_raw.csv data/main/weighted_summary.md
```
`analysis/verify_*.py`는 KASI 표와의 대조 검증입니다.

## 자료와 출처
위키데이터(CC0, 출생일·직업·위키백과 언어판 수, 2026-10-08 질의; 인물별 자료는 비공개), 한국천문연구원(2026) 24기 입기 시각(천문우주포털 역서, 2026-10-12; 표는 재배포하지 않음), 통계청(2026) 월.분기.연간 인구동향 전국 출생아수(KOSIS, 2026-10-12, `data/kosis_raw.csv`).

## 폴더
`index.html`(논문 웹 페이지), `paper/`(논문), `analysis/`(코드), `data/`(집계 결과)
