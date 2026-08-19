# Data Dictionary

Auto-generated from `Chilli mastersheet for dashboard.xlsx` on 2026-07-28 12:39:48.

- Worksheets in file: **13**
- Worksheets mapped to analyses: **13**
- Worksheets unmapped: **0**
- Parse time: **3.53s**

## Seasonality index_Guntur Teja

Guntur Teja monthly average price by calendar year, with the workbook's own multi-year average and seasonality index rows.

| Field | Role | Type | Rows | Populated | Minimum | Maximum | Unit |
|---|---|---|---|---|---|---|---|
| Jan | monthly average price | float64 | 13 | 100% | 7,850 | 1.997e+04 | INR/quintal |
| Feb | monthly average price | float64 | 13 | 100% | 7,708 | 2.003e+04 | INR/quintal |
| Mar | monthly average price | float64 | 13 | 100% | 7,519 | 2.213e+04 | INR/quintal |
| Apr | monthly average price | float64 | 13 | 92% | 6,433 | 2.108e+04 | INR/quintal |
| May | monthly average price | float64 | 13 | 100% | 5,000 | 2.103e+04 | INR/quintal |
| Jun | monthly average price | float64 | 13 | 100% | 5,764 | 2.357e+04 | INR/quintal |
| Jul | monthly average price | float64 | 13 | 92% | 7,279 | 2.399e+04 | INR/quintal |
| Aug | monthly average price | float64 | 13 | 92% | 7,603 | 2.37e+04 | INR/quintal |
| Sep | monthly average price | float64 | 13 | 85% | 9,441 | 2.379e+04 | INR/quintal |
| Oct | monthly average price | float64 | 13 | 92% | 8,089 | 2.295e+04 | INR/quintal |
| Nov | monthly average price | float64 | 13 | 92% | 9,027 | 2.277e+04 | INR/quintal |
| Dec | monthly average price | float64 | 13 | 92% | 8,686 | 2.223e+04 | INR/quintal |

## Guntur Varietywise daily price

Daily Guntur spot prices per variety. Two header rows: variety name spans four sub-columns (Low, High, Average, Difference).

| Field | Role | Type | Rows | Populated | Minimum | Maximum | Unit |
|---|---|---|---|---|---|---|---|
| date | variety average price | datetime64 | 2612 | 100% | 2014-01-10 | 2026-07-20 | INR/quintal |
| Teja | variety average price | float64 | 2612 | 100% | 4,500 | 2.45e+04 | INR/quintal |
| LCA 334 | variety average price | float64 | 2612 | 100% | 3,000 | 2.85e+04 | INR/quintal |
| NCDEX Quality | variety average price | float64 | 2612 | 100% | 2,750 | 2.65e+04 | INR/quintal |
| No.273 | variety average price | float64 | 2612 | 100% | 4,500 | 2.8e+04 | INR/quintal |
| No.5 | variety average price | float64 | 2612 | 100% | 3,250 | 2.9e+04 | INR/quintal |
| Fatki | variety average price | float64 | 2612 | 100% | 4,000 | 1.6e+04 | INR/quintal |
| Byadgi | variety average price | float64 | 2612 | 100% | 1,400 | 3.15e+04 | INR/quintal |
| US 341 | variety average price | float64 | 2612 | 100% | 3,500 | 3.025e+04 | INR/quintal |
| Denvor Delux | variety average price | float64 | 2612 | 100% | 5,000 | 2.95e+04 | INR/quintal |

## Guntur Variety correlation

Correlation matrix between Guntur varieties as supplied in the workbook. Displayed for reference alongside the values this application recomputes from the daily price sheet.

| Field | Role | Type | Rows | Populated | Minimum | Maximum | Unit |
|---|---|---|---|---|---|---|---|
| Teja | correlation coefficient | float64 | 9 | 100% | 0.5807 | 1 | dimensionless (-1 to +1) |
| LCA 334 | correlation coefficient | float64 | 9 | 100% | 0.6039 | 1 | dimensionless (-1 to +1) |
| Ncdex Quality | correlation coefficient | float64 | 9 | 100% | 0.6184 | 1 | dimensionless (-1 to +1) |
| No.273 | correlation coefficient | float64 | 9 | 100% | 0.5571 | 1 | dimensionless (-1 to +1) |
| No.5 | correlation coefficient | float64 | 9 | 100% | 0.5376 | 1 | dimensionless (-1 to +1) |
| Fatki | correlation coefficient | float64 | 9 | 100% | 0.4212 | 1 | dimensionless (-1 to +1) |
| Byadgi | correlation coefficient | float64 | 9 | 100% | 0.4212 | 1 | dimensionless (-1 to +1) |
| US 341 | correlation coefficient | float64 | 9 | 100% | 0.4798 | 1 | dimensionless (-1 to +1) |
| Denvor Delux | correlation coefficient | float64 | 9 | 100% | 0.5045 | 1 | dimensionless (-1 to +1) |
| Armoor | correlation coefficient | float64 | 9 | 100% | 0.9036 | 0.9418 | dimensionless (-1 to +1) |
| 5531 | correlation coefficient | float64 | 9 | 100% | 0.8153 | 0.9494 | dimensionless (-1 to +1) |
| Bangaram | correlation coefficient | float64 | 9 | 100% | -0.5139 | 0.5378 | dimensionless (-1 to +1) |
| 2043 Byadgi | correlation coefficient | float64 | 9 | 100% | 0.7144 | 0.9127 | dimensionless (-1 to +1) |
| Seed Fatki | correlation coefficient | float64 | 9 | 100% | 0.7635 | 0.8849 | dimensionless (-1 to +1) |
| 334 Fatki | correlation coefficient | float64 | 9 | 100% | 0.8505 | 0.9306 | dimensionless (-1 to +1) |
| Indam 5 | correlation coefficient | float64 | 9 | 100% | 0.8827 | 0.9049 | dimensionless (-1 to +1) |

## USD to INR exchange rate

Daily USD/INR reference rate.

| Field | Role | Type | Rows | Populated | Minimum | Maximum | Unit |
|---|---|---|---|---|---|---|---|
| date | date (index) | datetime64 | 2277 | 100% | 2014-01-01 | 2026-07-20 | — |
| USD | rate | float64 | 2277 | 100% | 58.43 | 96.84 | INR per USD |

## Guntur Daily arrivals

Guntur mandi daily arrivals and offtake, in bags.

| Field | Role | Type | Rows | Populated | Minimum | Maximum | Unit |
|---|---|---|---|---|---|---|---|
| date | date (index) | datetime64 | 2630 | 100% | 2014-01-10 | 2026-07-20 | — |
| Arrivals | arrivals | float64 | 2630 | 100% | 6,000 | 2e+05 | bags (1 bag = 45 kg, per sheet header) |
| Offtake | offtake | float64 | 2630 | 100% | 3,000 | 1.8e+05 | bags (1 bag = 45 kg, per sheet header) |

## Warangal Teja Price& Arrivals

Warangal Teja daily low/high/average price and arrivals.

| Field | Role | Type | Rows | Populated | Minimum | Maximum | Unit |
|---|---|---|---|---|---|---|---|
| date | date (index) | datetime64 | 2285 | 100% | 2015-03-12 | 2026-07-20 | — |
| low | low | float64 | 2285 | 100% | 3,500 | 2.3e+04 | INR/quintal |
| high | high | float64 | 2285 | 100% | 5,500 | 2.5e+04 | INR/quintal |
| average | average | float64 | 2285 | 100% | 4,500 | 2.375e+04 | INR/quintal |
| arrivals | arrivals | float64 | 2285 | 100% | 500 | 1.1e+05 | bags (1 bag = 40 kg, per sheet header) |

## Khammam Teja non cold storage

Khammam Teja daily price and arrivals for non-cold-storage (fresh) lots.

| Field | Role | Type | Rows | Populated | Minimum | Maximum | Unit |
|---|---|---|---|---|---|---|---|
| date | date (index) | datetime64 | 812 | 100% | 2017-06-13 | 2026-07-01 | — |
| low | low | float64 | 812 | 100% | 2,000 | 2.3e+04 | INR/quintal |
| high | high | float64 | 812 | 100% | 900 | 2.45e+04 | INR/quintal |
| average | average | float64 | 812 | 100% | 3,000 | 2.35e+04 | INR/quintal |
| arrivals | arrivals | float64 | 812 | 100% | 40 | 3e+05 | bags (1 bag = 40 kg, per sheet header) |

## KhammamTejacoldstorage

Khammam Teja daily price and arrivals for cold-storage lots.

| Field | Role | Type | Rows | Populated | Minimum | Maximum | Unit |
|---|---|---|---|---|---|---|---|
| date | date (index) | datetime64 | 1260 | 100% | 2017-06-13 | 2026-07-20 | — |
| low | low | float64 | 1260 | 100% | 6,000 | 2.4e+04 | INR/quintal |
| high | high | float64 | 1260 | 100% | 7,200 | 2.5e+04 | INR/quintal |
| average | average | float64 | 1260 | 100% | 6,600 | 2.425e+04 | INR/quintal |
| arrivals | arrivals | float64 | 1260 | 100% | 70 | 7.5e+04 | bags (1 bag = 40 kg, per sheet header) |

## Red chilli exports

Red chilli exports by calendar month and year.

| Field | Role | Type | Rows | Populated | Minimum | Maximum | Unit |
|---|---|---|---|---|---|---|---|
| Jan | monthly exports | float64 | 12 | 100% | 2.084e+04 | 4.045e+04 | as supplied (unit not stated on sheet) |
| Feb | monthly exports | float64 | 12 | 100% | 2.848e+04 | 9.747e+04 | as supplied (unit not stated on sheet) |
| Mar | monthly exports | float64 | 12 | 100% | 3.522e+04 | 1.212e+05 | as supplied (unit not stated on sheet) |
| Apr | monthly exports | float64 | 12 | 100% | 2.964e+04 | 1.131e+05 | as supplied (unit not stated on sheet) |
| May | monthly exports | float64 | 12 | 100% | 2.454e+04 | 6.113e+04 | as supplied (unit not stated on sheet) |
| Jun | monthly exports | float64 | 12 | 92% | 1.782e+04 | 3.907e+04 | as supplied (unit not stated on sheet) |
| Jul | monthly exports | float64 | 12 | 92% | 2.404e+04 | 6.165e+04 | as supplied (unit not stated on sheet) |
| Aug | monthly exports | float64 | 12 | 92% | 2.598e+04 | 4.614e+04 | as supplied (unit not stated on sheet) |
| Sep | monthly exports | float64 | 12 | 92% | 2.383e+04 | 5.644e+04 | as supplied (unit not stated on sheet) |
| Oct | monthly exports | float64 | 12 | 92% | 2.015e+04 | 5.501e+04 | as supplied (unit not stated on sheet) |
| Nov | monthly exports | float64 | 12 | 92% | 1.933e+04 | 5.589e+04 | as supplied (unit not stated on sheet) |
| Dec | monthly exports | float64 | 12 | 92% | 2.014e+04 | 5.009e+04 | as supplied (unit not stated on sheet) |

## Red Chilli Balance sheet

National red chilli supply/demand balance sheet by calendar year, in lakh tonnes.

| Field | Role | Type | Rows | Populated | Minimum | Maximum | Unit |
|---|---|---|---|---|---|---|---|
| 2017 | historical | float64 | 9 | 100% | 0 | 33.59 | Unit - Lakh Tons |
| 2018 | historical | float64 | 9 | 100% | 0 | 19.53 | Unit - Lakh Tons |
| 2019 | historical | float64 | 9 | 100% | 0 | 14.65 | Unit - Lakh Tons |
| 2020 | historical | float64 | 9 | 100% | 0 | 14 | Unit - Lakh Tons |
| 2021 | historical | float64 | 9 | 100% | 0 | 14.54 | Unit - Lakh Tons |
| 2022 | historical | float64 | 9 | 100% | 0 | 16.93 | Unit - Lakh Tons |
| 2023 | historical | float64 | 9 | 100% | 0 | 30.55 | Unit - Lakh Tons |
| 2024 | historical | float64 | 9 | 100% | 0 | 52.02 | Unit - Lakh Tons |
| 2025 | historical | float64 | 9 | 100% | 0 | 24.44 | Unit - Lakh Tons |
| 2026 | projection | float64 | 9 | 100% | 0 | 13.05 | Unit - Lakh Tons |

## Guntur Monthly Arrivals

Guntur monthly arrivals, supplied twice: once in bags and once in the workbook's converted unit.

| Field | Role | Type | Rows | Populated | Minimum | Maximum | Unit |
|---|---|---|---|---|---|---|---|
| Jan | monthly arrivals | float64 | 13 | 100% | 5.38e+05 | 2.145e+06 | bags (1 bag = 45 kg, per sheet annotation) |
| Feb | monthly arrivals | float64 | 13 | 100% | 8.25e+05 | 2.675e+06 | bags (1 bag = 45 kg, per sheet annotation) |
| Mar | monthly arrivals | float64 | 13 | 100% | 1.12e+06 | 2.395e+06 | bags (1 bag = 45 kg, per sheet annotation) |
| Apr | monthly arrivals | float64 | 13 | 92% | 8.85e+05 | 2.365e+06 | bags (1 bag = 45 kg, per sheet annotation) |
| May | monthly arrivals | float64 | 13 | 100% | 1.3e+05 | 8.15e+05 | bags (1 bag = 45 kg, per sheet annotation) |
| Jun | monthly arrivals | float64 | 13 | 100% | 5.1e+05 | 9.3e+05 | bags (1 bag = 45 kg, per sheet annotation) |
| Jul | monthly arrivals | float64 | 13 | 92% | 4.25e+05 | 1.47e+06 | bags (1 bag = 45 kg, per sheet annotation) |
| Aug | monthly arrivals | float64 | 13 | 92% | 7.95e+05 | 1.38e+06 | bags (1 bag = 45 kg, per sheet annotation) |
| Sep | monthly arrivals | float64 | 13 | 85% | 7.65e+05 | 1.68e+06 | bags (1 bag = 45 kg, per sheet annotation) |
| Oct | monthly arrivals | float64 | 13 | 92% | 5.75e+05 | 1.85e+06 | bags (1 bag = 45 kg, per sheet annotation) |
| Nov | monthly arrivals | float64 | 13 | 92% | 6.25e+05 | 1.996e+06 | bags (1 bag = 45 kg, per sheet annotation) |
| Dec | monthly arrivals | float64 | 13 | 92% | 7.07e+05 | 1.591e+06 | bags (1 bag = 45 kg, per sheet annotation) |

## APY

Area, production and yield by state and year (APY statistics).

| Field | Role | Type | Rows | Populated | Minimum | Maximum | Unit |
|---|---|---|---|---|---|---|---|
| state | unknown | str | 104 | 100% | Andhra Pradesh | Telangana | — |
| year | date | int64 | 104 | 100% | 2,014 | 2,026 | — |
| Area (Ha) | unknown | float64 | 104 | 100% | 5,073 | 1.047e+06 | — |
| Production (MT) | unknown | float64 | 104 | 92% | 3,957 | 1.769e+06 | — |
| Yield (t/Ha) | unknown | float64 | 104 | 92% | 0.4 | 3.9 | — |

## cold storage

Reported cold storage stock positions by state/market and month.

| Field | Role | Type | Rows | Populated | Minimum | Maximum | Unit |
|---|---|---|---|---|---|---|---|
| date | date (index) | datetime64 | 6 | 100% | 2025-04-01 | 2026-07-01 | — |
| AP Stock (Bags) | unknown | float64 | 6 | 33% | 4.5e+06 | 1.68e+07 | bags |
| Guntur Stock (Bags) | unknown | float64 | 6 | 33% | 3.277e+06 | 6.956e+06 | bags |
| Karnataka Stock (Bags) | unknown | float64 | 6 | 33% | 2.6e+06 | 8.5e+06 | bags |
| Khammam Stock (Bags) | unknown | float64 | 6 | 50% | 1.4e+06 | 1.645e+06 | bags |
| Telangana Stock (Bags) | unknown | float64 | 6 | 33% | 3.5e+06 | 8.1e+06 | bags |
| Warangal Stock (Bags) | unknown | float64 | 6 | 33% | 1.767e+06 | 1.811e+06 | bags |

## Parse warnings

- cold storage: Only 6 reporting month(s) are present; the densest location has 3 observation(s). This is far below what any time-series or correlation analysis requires.
