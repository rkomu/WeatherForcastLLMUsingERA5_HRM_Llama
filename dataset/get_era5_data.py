from configure import configure
from get_era5 import ERA5

# 設定値を取得する関数

import concurrent.futures

dt1, dt2, lev, dir = configure()
print(f'{dt1}/{dt2}/{lev}')

# ERA5からデータを取得する関数
ERA5()

################################################################################
# 各データをダウンロードしてCSVファイルを生成 (並列化)
################################################################################
def download_tasks():
    tasks = []
    # ERA5 hourly data on pressure levels
    tasks.append((ERA5.reanalysis_era5_pressure_levels, ('temperature', 't', lev, dt1, dt2, dir)))
    tasks.append((ERA5.reanalysis_era5_pressure_levels, ('relative_humidity', 'r', lev, dt1, dt2, dir)))
    # ERA5 hourly data on single levels
    tasks.append((ERA5.reanalysis_era5_single_levels, ('surface_pressure', 'sp', dt1, dt2, dir)))
    tasks.append((ERA5.reanalysis_era5_single_levels, ('10m_u_component_of_wind', 'u10', dt1, dt2, dir)))
    tasks.append((ERA5.reanalysis_era5_single_levels, ('10m_v_component_of_wind', 'v10', dt1, dt2, dir)))
    tasks.append((ERA5.reanalysis_era5_single_levels, ('convective_precipitation', 'cp', dt1, dt2, dir)))
    tasks.append((ERA5.reanalysis_era5_single_levels, ('surface_solar_radiation_downwards', 'ssrd', dt1, dt2, dir)))
    return tasks

def run_parallel_downloads():
    tasks = download_tasks()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(func, *args) for func, args in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Download failed: {e}")

if __name__ == "__main__":
    run_parallel_downloads()
