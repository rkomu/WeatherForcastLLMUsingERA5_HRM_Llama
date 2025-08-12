from configure import configure
from get_era5 import ERA5

# 設定値を取得する関数
dt1, dt2, lev, dir = configure()
print(f'{dt1}/{dt2}/{lev}')

# ERA5からデータを取得する関数
ERA5()

################################################################################
# 各データをダウンロードしてCSVファイルを生成
################################################################################
#-------------------------------------------------------------------------------
# ERA5 hourly data on pressure levels from 1940 to present
#-------------------------------------------------------------------------------
# 気温 (K)
ERA5.reanalysis_era5_pressure_levels('temperature', 't', lev, dt1, dt2, dir)

# 相対湿度 (%RH)
ERA5.reanalysis_era5_pressure_levels('relative_humidity', 'r', lev, dt1, dt2, dir)

#-------------------------------------------------------------------------------
# ERA5 hourly data on single levels from 1940 to present
#-------------------------------------------------------------------------------
# 気圧 (Pa)
ERA5.reanalysis_era5_single_levels('surface_pressure', 'sp', dt1, dt2, dir)

# 風速 (m/s)
ERA5.reanalysis_era5_single_levels('10m_u_component_of_wind', 'u10', dt1, dt2, dir)
ERA5.reanalysis_era5_single_levels('10m_v_component_of_wind', 'v10', dt1, dt2, dir)

# 降水量 (m/h)
ERA5.reanalysis_era5_single_levels('convective_precipitation', 'cp', dt1, dt2, dir)

# 全天日射量 (J/m2)
ERA5.reanalysis_era5_single_levels('surface_solar_radiation_downwards', 'ssrd', dt1, dt2, dir)
