# Detailed results

All numbers are computed over the held-out 2019-2020 ERA5 validation years, autoregressively rolled out to 60 steps (15 days) from the `ar6` checkpoint. RMSE is latitude-weighted and reported in physical units; ACC is latitude-weighted anomaly correlation against the WeatherBench 2 1990-2019 hourly climatology.

Regenerate this file with `python scripts/summarize_results.py`.


## Anomaly correlation coefficient (48-layer)

| Variable | 1 d | 2 d | 3 d | 5 d | 7 d | 10 d | 15 d | Skill horizon (ACC > 0.6) |
|---|---|---|---|---|---|---|---|---|
| `2m_temperature` | 0.924 | 0.883 | 0.837 | 0.702 | 0.522 | 0.293 | 0.089 | 6.00 d |
| `10m_u_component_of_wind` | 0.966 | 0.926 | 0.862 | 0.672 | 0.461 | 0.229 | 0.075 | 5.50 d |
| `10m_v_component_of_wind` | 0.966 | 0.926 | 0.862 | 0.667 | 0.440 | 0.197 | 0.050 | 5.50 d |
| `mean_sea_level_pressure` | 1.000 | 1.000 | 0.999 | 0.998 | 0.996 | 0.994 | 0.991 | > 15 d |
| `total_column_water_vapour` | 0.951 | 0.896 | 0.827 | 0.662 | 0.486 | 0.273 | 0.102 | 5.50 d |
| `temperature_850` | 0.966 | 0.940 | 0.901 | 0.767 | 0.573 | 0.307 | 0.087 | 6.50 d |
| `geopotential_850` | 0.991 | 0.974 | 0.943 | 0.818 | 0.627 | 0.356 | 0.118 | 7.25 d |
| `specific_humidity_850` | 0.896 | 0.811 | 0.724 | 0.553 | 0.395 | 0.218 | 0.083 | 4.25 d |
| `u_component_of_wind_850` | 0.963 | 0.924 | 0.864 | 0.687 | 0.485 | 0.250 | 0.087 | 5.75 d |
| `v_component_of_wind_850` | 0.960 | 0.919 | 0.855 | 0.662 | 0.438 | 0.202 | 0.050 | 5.50 d |
| `temperature_500` | 0.976 | 0.951 | 0.910 | 0.766 | 0.564 | 0.296 | 0.079 | 6.50 d |
| `geopotential_500` | 0.994 | 0.980 | 0.954 | 0.844 | 0.661 | 0.382 | 0.123 | 7.50 d |
| `specific_humidity_500` | 0.910 | 0.826 | 0.739 | 0.569 | 0.417 | 0.245 | 0.096 | 4.50 d |
| `u_component_of_wind_500` | 0.969 | 0.934 | 0.882 | 0.731 | 0.544 | 0.300 | 0.109 | 6.25 d |
| `v_component_of_wind_500` | 0.967 | 0.931 | 0.877 | 0.706 | 0.491 | 0.233 | 0.059 | 5.75 d |
| `temperature_250` | 0.961 | 0.913 | 0.847 | 0.668 | 0.479 | 0.265 | 0.098 | 5.50 d |
| `geopotential_250` | 0.994 | 0.981 | 0.956 | 0.851 | 0.671 | 0.382 | 0.109 | 7.50 d |
| `specific_humidity_250` | 0.845 | 0.746 | 0.663 | 0.520 | 0.389 | 0.228 | 0.085 | 3.75 d |
| `u_component_of_wind_250` | 0.974 | 0.943 | 0.899 | 0.761 | 0.580 | 0.332 | 0.130 | 6.75 d |
| `v_component_of_wind_250` | 0.973 | 0.943 | 0.896 | 0.740 | 0.526 | 0.249 | 0.066 | 6.25 d |
| **mean (19 vars)** | **0.955** | **0.913** | **0.858** | **0.702** | **0.513** | **0.276** | **0.089** | |

`mean_sea_level_pressure` is reported for completeness but excluded from the mean: the climatology zarr has no MSLP field and the evaluation falls back to `surface_pressure`, which inflates its ACC. See Known issues in the README.


## Latitude-weighted RMSE (48-layer, physical units)

| Variable | Unit | 1 d | 2 d | 3 d | 5 d | 7 d | 10 d | 15 d |
|---|---|---|---|---|---|---|---|---|
| `2m_temperature` | K | 0.975 | 1.21 | 1.42 | 1.92 | 2.44 | 3.02 | 3.53 |
| `10m_u_component_of_wind` | m/s | 0.983 | 1.44 | 1.95 | 2.99 | 3.83 | 4.6 | 5.06 |
| `10m_v_component_of_wind` | m/s | 1.02 | 1.49 | 2.02 | 3.11 | 4.02 | 4.83 | 5.27 |
| `mean_sea_level_pressure` | Pa | 98.3 | 165 | 244 | 432 | 615 | 805 | 938 |
| `total_column_water_vapour` | kg/m^2 | 1.95 | 2.82 | 3.61 | 5.01 | 6.14 | 7.3 | 8.13 |
| `temperature_850` | K | 0.88 | 1.16 | 1.48 | 2.26 | 3.05 | 3.9 | 4.54 |
| `geopotential_850` | m^2/s^2 | 72.8 | 124 | 183 | 324 | 464 | 615 | 723 |
| `specific_humidity_850` | kg/kg | 0.000807 | 0.00107 | 0.00128 | 0.00161 | 0.00186 | 0.0021 | 0.00228 |
| `u_component_of_wind_850` | m/s | 1.5 | 2.15 | 2.85 | 4.28 | 5.48 | 6.65 | 7.37 |
| `v_component_of_wind_850` | m/s | 1.53 | 2.16 | 2.86 | 4.31 | 5.53 | 6.61 | 7.23 |
| `temperature_500` | K | 0.684 | 0.969 | 1.31 | 2.1 | 2.86 | 3.67 | 4.26 |
| `geopotential_500` | m^2/s^2 | 88.6 | 158 | 242 | 442 | 649 | 881 | 1058 |
| `specific_humidity_500` | kg/kg | 0.000347 | 0.000474 | 0.000574 | 0.000728 | 0.000841 | 0.000951 | 0.00103 |
| `u_component_of_wind_500` | m/s | 2.04 | 2.96 | 3.93 | 5.89 | 7.66 | 9.5 | 10.8 |
| `v_component_of_wind_500` | m/s | 2.1 | 3.02 | 4.01 | 6.13 | 8.03 | 9.89 | 11 |
| `temperature_250` | K | 0.838 | 1.24 | 1.62 | 2.36 | 2.94 | 3.49 | 3.9 |
| `geopotential_250` | m^2/s^2 | 125 | 220 | 333 | 606 | 900 | 1240 | 1511 |
| `specific_humidity_250` | kg/kg | 3.06e-05 | 3.83e-05 | 4.35e-05 | 5.09e-05 | 5.66e-05 | 6.27e-05 | 6.69e-05 |
| `u_component_of_wind_250` | m/s | 2.84 | 4.14 | 5.52 | 8.48 | 11.3 | 14.3 | 16.4 |
| `v_component_of_wind_250` | m/s | 2.9 | 4.23 | 5.68 | 8.94 | 12.1 | 15.2 | 17 |

## Depth ablation: 48 layers vs 6 layers

Identical data, graph, optimiser and curriculum; only `num_bottleneck_layers` differs.

| Variable | RMSE 3 d (48L) | RMSE 3 d (6L) | ACC 3 d (48L) | ACC 3 d (6L) | Skill horizon 48L | Skill horizon 6L |
|---|---|---|---|---|---|---|
| `2m_temperature` | 1.42 | 1.77 | 0.837 | 0.757 | 6.00 d | 4.75 d |
| `10m_u_component_of_wind` | 1.95 | 2.21 | 0.862 | 0.820 | 5.50 d | 5.00 d |
| `10m_v_component_of_wind` | 2.02 | 2.28 | 0.862 | 0.821 | 5.50 d | 4.75 d |
| `mean_sea_level_pressure` | 244 | 300 | 0.999 | 0.999 | > 15 d | > 15 d |
| `total_column_water_vapour` | 3.61 | 4 | 0.827 | 0.784 | 5.50 d | 4.75 d |
| `temperature_850` | 1.48 | 1.73 | 0.901 | 0.864 | 6.50 d | 5.75 d |
| `geopotential_850` | 183 | 227 | 0.943 | 0.912 | 7.25 d | 6.50 d |
| `specific_humidity_850` | 0.00128 | 0.00138 | 0.724 | 0.671 | 4.25 d | 3.50 d |
| `u_component_of_wind_850` | 2.85 | 3.22 | 0.864 | 0.824 | 5.75 d | 5.00 d |
| `v_component_of_wind_850` | 2.86 | 3.22 | 0.855 | 0.813 | 5.50 d | 4.75 d |
| `temperature_500` | 1.31 | 1.51 | 0.910 | 0.879 | 6.50 d | 6.00 d |
| `geopotential_500` | 242 | 304 | 0.954 | 0.927 | 7.50 d | 6.75 d |
| `specific_humidity_500` | 0.000574 | 0.000612 | 0.739 | 0.697 | 4.50 d | 3.75 d |
| `u_component_of_wind_500` | 3.93 | 4.44 | 0.882 | 0.847 | 6.25 d | 5.75 d |
| `v_component_of_wind_500` | 4.01 | 4.53 | 0.877 | 0.841 | 5.75 d | 5.25 d |
| `temperature_250` | 1.62 | 1.84 | 0.847 | 0.798 | 5.50 d | 4.75 d |
| `geopotential_250` | 333 | 420 | 0.956 | 0.929 | 7.50 d | 6.75 d |
| `specific_humidity_250` | 4.35e-05 | 4.58e-05 | 0.663 | 0.616 | 3.75 d | 3.00 d |
| `u_component_of_wind_250` | 5.52 | 6.3 | 0.899 | 0.867 | 6.75 d | 6.00 d |
| `v_component_of_wind_250` | 5.68 | 6.46 | 0.896 | 0.864 | 6.25 d | 5.50 d |
| **mean ACC @ 1 d** | | | **0.955** | **0.943** | | |
| **mean ACC @ 3 d** | | | **0.858** | **0.817** | | |
| **mean ACC @ 5 d** | | | **0.702** | **0.637** | | |
| **mean ACC @ 7 d** | | | **0.513** | **0.445** | | |

## Spectral energy retention (48-layer)

Ratio of predicted to ERA5 zonal power, averaged over synoptic (wavenumber 20-59) and sub-synoptic (60-120) bands. 1.0 means the forecast carries the same variance as the truth at those scales.

| Lead | `2m_temperature` syn / sub | `temperature_850` syn / sub | `u_component_of_wind_250` syn / sub |
|---|---|---|---|
| 6 h | 0.85 / 0.79 | 0.63 / 0.36 | 0.48 / 0.25 |
| 30 h | 0.80 / 0.74 | 0.50 / 0.28 | 0.28 / 0.18 |
| 60 h | 0.79 / 0.74 | 0.47 / 0.27 | 0.25 / 0.17 |
| 120 h | 0.79 / 0.74 | 0.45 / 0.27 | 0.23 / 0.16 |
| 240 h | 0.79 / 0.74 | 0.45 / 0.26 | 0.23 / 0.16 |
| 360 h | 0.80 / 0.74 | 0.44 / 0.26 | 0.23 / 0.16 |
