import cdsapi
import netCDF4
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from datetime import date, datetime, timedelta
from geopy.geocoders import Nominatim
from tqdm import tqdm
import time

import certifi
import ssl
import urllib3

class ERA5():
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    urllib3.util.ssl_.create_urllib3_context = lambda *args, **kwargs: ERA5.ssl_context

    ################################################################################
    # データ抽出期間を補正する関数
    ################################################################################
    def correct_dt(shortname, dt1, dt2, path, files):
        filelist = []
        
        for filename in files:
            if os.path.isfile(os.path.join(path, filename)):
                filelist.append(filename)
        
        # データ取得開始日時
        if len(filelist) == 0:
            start = 'ERA5_'+str(format(dt1.year, '04'))+'-'+str(format(dt1.month, '02'))+'-'+str(format(dt1.day, '02'))+'T'+str(format(dt1.hour, '02'))+'_00_00_'+str(shortname)+'.nc'
        else:
            start = max(filelist)
        str_d1 = str(start[5:9])+str(start[10:12])+str(start[13:15])+str(start[16:18])
        
        # データ取得終了日時
        if datetime.now() - timedelta(7) > dt2:
            end = dt2
        else:
            end = datetime.now() - timedelta(7)
        str_d2 = end.strftime('%Y%m%d%H')
        
        # データ抽出期間を設定
        dt1 = datetime(int(str_d1[:4]), int(str_d1[4:6]), int(str_d1[6:8]), int(str_d1[8:]), 0, 0)
        dt2 = datetime(int(str_d2[:4]), int(str_d2[4:6]), int(str_d2[6:8]), int(str_d2[8:]), 0, 0)

    def correct_dt_single(shortname, dt1, dt2, dir):
        if not os.path.exists(str(dir)+'/nc_'+str(shortname)):
            os.makedirs(str(dir)+'/nc_'+str(shortname))
            
        path = str(dir)+'/nc_'+str(shortname)
        files = os.listdir(path)

        ERA5.correct_dt(shortname, dt1, dt2, path, files)

        return (dt1, dt2, dir)

    def correct_dt_pressure(shortname, lev, dt1, dt2, dir):
        if not os.path.exists(str(dir)+'/nc_'+str(shortname)):
            os.makedirs(str(dir)+'/nc_'+str(shortname))
            
        path = str(dir)+'/nc_'+str(shortname)
        files = os.listdir(path)

        ERA5.correct_dt(shortname, dt1, dt2, path, files)

        return (dt1, dt2, dir)

    ################################################################################
    # 「ERA5 hourly data on single levels from 1940 to present」からデータを抽出する関数
    # https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels
    ################################################################################
    def reanalysis_era5_single_levels(name, shortname, dt1, dt2, dir):
        # データ抽出期間を補正
        period = ERA5.correct_dt_single(shortname, dt1, dt2, dir)
        dt1 = period[0]
        dt2 = period[1]
        print(f'{shortname}: {dt1} - {dt2}')

        dt = dt1

        if not os.path.exists(str(dir)+'/nc_'+str(shortname)):
            os.makedirs(str(dir)+'/nc_'+str(shortname))

        while dt <= dt2:
            print(str(name))
            print(dt)

            ncfile = str(dir)+'/nc_'+str(shortname)+'/ERA5_'+str(format(dt.year, '04'))+'-'+str(format(dt.month, '02'))+'-'+str(format(dt.day, '02'))+'T'+str(format(dt.hour, '02'))+'_00_00_'+str(shortname)+'.nc'
            
            if os.path.isfile(ncfile) is False:
                ERA5.c.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'product_type': 'reanalysis',
                        'variable': str(name),
                        'year': str(dt.year),
                        'month': str(dt.month),
                        'day': str(dt.day),
                        'valid_time': str(dt.strftime('%H:%M')),
                        'format': 'netcdf'
                    },
                    str(ncfile))
            else:
                pass

            dt += ERA5.delta

    ################################################################################
    # 「ERA5 hourly data on pressure levels from 1940 to present」からデータを抽出する関数
    # https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels
    ################################################################################
    def reanalysis_era5_pressure_levels(name, shortname, lev, dt1, dt2, dir):
        # データ抽出期間を補正
        period = ERA5.correct_dt_pressure(shortname, lev, dt1, dt2, dir)
        dt1 = period[0]
        dt2 = period[1]
        print(f'{shortname}: {dt1} - {dt2}')
        
        dt = dt1

        if not os.path.exists(str(dir)+'/nc_'+str(shortname)):
            os.makedirs(str(dir)+'/nc_'+str(shortname))

        while dt <= dt2:
            print(str(name))
            print(dt)

            ncfile = str(dir)+'/nc_'+str(shortname)+'/ERA5_'+str(format(dt.year, '04'))+'-'+str(format(dt.month, '02'))+'-'+str(format(dt.day, '02'))+'T'+str(format(dt.hour, '02'))+'_00_00_'+str(shortname)+'.nc'

            if os.path.isfile(ncfile) is False:
                ERA5.c.retrieve(
                    'reanalysis-era5-pressure-levels',
                    {
                        'product_type': 'reanalysis',
                        'variable': str(name),
                        'year': str(dt.year),
                        'month': str(dt.month),
                        'day': str(dt.day),
                        'valid_time': str(dt.strftime('%H:%M')),
                        'format': 'netcdf'
                    },
                    str(ncfile))
            else:
                pass
                
            dt += ERA5.delta

    ################################################################################
    # 変数
    ################################################################################
    # 時間間隔
    delta = timedelta(hours=1)

    # CDS API
    c = cdsapi.Client()
