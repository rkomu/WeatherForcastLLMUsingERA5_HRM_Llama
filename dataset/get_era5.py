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

import logging
import traceback
import re

# Setup module logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

class ERA5():
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    urllib3.util.ssl_.create_urllib3_context = lambda *args, **kwargs: ERA5.ssl_context

    ################################################################################
    # ヘルパー: 期待される日時リストを作成
    ################################################################################
    def _expected_datetimes(dt1, dt2):
        dts = []
        dt = dt1
        while dt <= dt2:
            dts.append(dt)
            dt += ERA5.delta
        return dts

    ################################################################################
    # ヘルパー: ディレクトリ内の既存ファイルから日時をパースして集合で返す
    ################################################################################
    def _existing_datetimes(path, shortname):
        import re
        pattern = re.compile(rf"ERA5_(\d{{4}})-(\d{{2}})-(\d{{2}})T(\d{{2}})_00_00_{re.escape(str(shortname))}\.nc$")
        existing = set()
        try:
            filenames = os.listdir(path)
            logger.debug("_existing_datetimes: scanning %s, total files=%d", path, len(filenames))
            sample = filenames[:10]
            if len(sample) > 0:
                logger.debug("_existing_datetimes: sample filenames=%s", sample)
            for filename in filenames:
                m = pattern.search(filename)
                if m:
                    y, mo, d, h = m.groups()
                    try:
                        existing.add(datetime(int(y), int(mo), int(d), int(h), 0, 0))
                    except Exception:
                        logger.debug("_existing_datetimes: failed to parse datetime from %s", filename)
        except FileNotFoundError:
            logger.debug("_existing_datetimes: path not found %s", path)
        logger.debug("_existing_datetimes: parsed %d existing timestamps for %s", len(existing), shortname)
        return existing

    ################################################################################
    # データ抽出期間を補正する関数
    ################################################################################
    def correct_dt(shortname, dt1, dt2, path, files):
        filelist = []
        
        for filename in files:
            if os.path.isfile(os.path.join(path, filename)):
                filelist.append(filename)
        logger.debug("correct_dt: scanning path %s found %d files", path, len(filelist))
        if len(filelist) > 0:
            # log a small sample and the earliest/latest files alphabetically (which match naming)
            sample = filelist[:5]
            logger.debug("correct_dt: sample files=%s", sample)
            try:
                earliest = min(filelist)
                latest = max(filelist)
                logger.debug("correct_dt: earliest=%s latest=%s", earliest, latest)
            except Exception:
                pass
        
        # NOTE: do not shift the requested start (dt1) to the latest existing file here.
        # The previous behavior used the latest filename as the new start which prevented
        # detecting and downloading missing files earlier in the requested period.
        # Keep dt1 as requested by the caller and only adjust the end (dt2) to be at
        # most now()-7days as required by the CDS availability.
        str_d1 = dt1.strftime('%Y%m%d%H')
        logger.debug("correct_dt: using requested start %s -> %s", dt1, str_d1)
        
        # データ取得終了日時 (cap to now()-7days)
        if datetime.now() - timedelta(7) > dt2:
            end = dt2
        else:
            end = datetime.now() - timedelta(7)
        str_d2 = end.strftime('%Y%m%d%H')
        logger.debug("correct_dt: using end %s -> %s", end, str_d2)
        
        # データ抽出期間を設定 (use the original dt1, but normalized to hour boundaries)
        dt1_adj = datetime(int(str_d1[:4]), int(str_d1[4:6]), int(str_d1[6:8]), int(str_d1[8:]), 0, 0)
        dt2_adj = datetime(int(str_d2[:4]), int(str_d2[4:6]), int(str_d2[6:8]), int(str_d2[8:]), 0, 0)
        logger.info("correct_dt: adjusted period %s - %s", dt1_adj, dt2_adj)
        return (dt1_adj, dt2_adj)

    def correct_dt_single(shortname, dt1, dt2, dir):
        if not os.path.exists(str(dir)+'/nc_'+str(shortname)):
            os.makedirs(str(dir)+'/nc_'+str(shortname))
            logger.info("correct_dt_single: created directory %s", str(dir)+'/nc_'+str(shortname))
        else:
            logger.debug("correct_dt_single: directory exists %s", str(dir)+'/nc_'+str(shortname))
            
        path = str(dir)+'/nc_'+str(shortname)
        files = os.listdir(path)

        logger.debug("correct_dt_single: calling correct_dt for %s", shortname)
        dt1_adj, dt2_adj = ERA5.correct_dt(shortname, dt1, dt2, path, files)

        logger.info("correct_dt_single: finished for %s -> %s - %s", shortname, dt1_adj, dt2_adj)
        return (dt1_adj, dt2_adj, dir)

    def correct_dt_pressure(shortname, lev, dt1, dt2, dir):
        if not os.path.exists(str(dir)+'/nc_'+str(shortname)):
            os.makedirs(str(dir)+'/nc_'+str(shortname))
            logger.info("correct_dt_pressure: created directory %s", str(dir)+'/nc_'+str(shortname))
        else:
            logger.debug("correct_dt_pressure: directory exists %s", str(dir)+'/nc_'+str(shortname))
            
        path = str(dir)+'/nc_'+str(shortname)
        files = os.listdir(path)

        logger.debug("correct_dt_pressure: calling correct_dt for %s level %s", shortname, lev)
        dt1_adj, dt2_adj = ERA5.correct_dt(shortname, dt1, dt2, path, files)

        logger.info("correct_dt_pressure: finished for %s level %s -> %s - %s", shortname, lev, dt1_adj, dt2_adj)
        return (dt1_adj, dt2_adj, dir)

    ################################################################################
    # 「ERA5 hourly data on single levels from 1940 to present」からデータを抽出する関数
    # https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels
    ################################################################################
    def reanalysis_era5_single_levels(name, shortname, dt1, dt2, dir):
        import concurrent.futures
        # データ抽出期間を補正
        period = ERA5.correct_dt_single(shortname, dt1, dt2, dir)
        dt1 = period[0]
        dt2 = period[1]
        logger.info("%s: %s - %s", shortname, dt1, dt2)

        if not os.path.exists(str(dir)+'/nc_'+str(shortname)):
            os.makedirs(str(dir)+'/nc_'+str(shortname))
            logger.debug("reanalysis_era5_single_levels: created directory %s", str(dir)+'/nc_'+str(shortname))

        path = str(dir)+'/nc_'+str(shortname)

        # build expected and existing datetime sets and find missing
        expected = set(ERA5._expected_datetimes(dt1, dt2))
        existing = ERA5._existing_datetimes(path, shortname)
        missing = sorted(list(expected - existing))

        logger.info("reanalysis_era5_single_levels: expected=%d existing=%d missing=%d", len(expected), len(existing), len(missing))
        if len(missing) > 0:
            logger.debug("reanalysis_era5_single_levels: first_missing=%s last_missing=%s", missing[0], missing[-1])

        if len(missing) == 0:
            logger.info("reanalysis_era5_single_levels: no missing files for %s", shortname)
            return

        def download(dt):
            logger.info("Starting download for %s at %s", shortname, dt)
            logger.debug("download: variable=%s, dt=%s", name, dt)
            ncfile = str(dir)+'/nc_'+str(shortname)+'/ERA5_'+str(format(dt.year, '04'))+'-'+str(format(dt.month, '02'))+'-'+str(format(dt.day, '02'))+'T'+str(format(dt.hour, '02'))+'_00_00_'+str(shortname)+'.nc'
            try:
                # double-check file existence in case of race conditions
                if os.path.isfile(ncfile):
                    logger.debug("download: file already present (race) %s", ncfile)
                    logger.info("Skipped download (already present): %s", ncfile)
                    return
                logger.info("Requesting file from CDS: %s", ncfile)
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
                logger.info("Saved file: %s", ncfile)
                logger.info("Finished download for %s at %s", shortname, dt)
            except Exception as e:
                logger.exception("Failed to download %s for %s: %s", name, dt, e)
                logger.debug(traceback.format_exc())

        max_workers = 8
        logger.info("Starting ThreadPoolExecutor with max_workers=%d for %s (missing=%d)", max_workers, shortname, len(missing))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(download, d) for d in missing]
            for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Downloading {shortname}")):
                try:
                    future.result()
                    if (i+1) % 10 == 0:
                        logger.info("Progress: %d/%d downloads completed for %s", i+1, len(futures), shortname)
                except Exception as e:
                    logger.exception("Download failed in executor: %s", e)

    ################################################################################
    # 「ERA5 hourly data on pressure levels from 1940 to present」からデータを抽出する関数
    # https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels
    ################################################################################
    def reanalysis_era5_pressure_levels(name, shortname, lev, dt1, dt2, dir):
        import concurrent.futures
        # データ抽出期間を補正
        period = ERA5.correct_dt_pressure(shortname, lev, dt1, dt2, dir)
        dt1 = period[0]
        dt2 = period[1]
        logger.info("%s (lev=%s): %s - %s", shortname, lev, dt1, dt2)

        if not os.path.exists(str(dir)+'/nc_'+str(shortname)):
            os.makedirs(str(dir)+'/nc_'+str(shortname))
            logger.debug("reanalysis_era5_pressure_levels: created directory %s", str(dir)+'/nc_'+str(shortname))

        path = str(dir)+'/nc_'+str(shortname)

        # build expected and existing datetime sets and find missing
        expected = set(ERA5._expected_datetimes(dt1, dt2))
        existing = ERA5._existing_datetimes(path, shortname)
        missing = sorted(list(expected - existing))

        logger.info("reanalysis_era5_pressure_levels: expected=%d existing=%d missing=%d", len(expected), len(existing), len(missing))
        if len(missing) > 0:
            logger.debug("reanalysis_era5_pressure_levels: first_missing=%s last_missing=%s", missing[0], missing[-1])

        if len(missing) == 0:
            logger.info("reanalysis_era5_pressure_levels: no missing files for %s lev=%s", shortname, lev)
            return

        def download(dt):
            logger.info("Starting pressure download for %s lev=%s at %s", shortname, lev, dt)
            logger.debug("download pressure: variable=%s level=%s dt=%s", name, lev, dt)
            ncfile = str(dir)+'/nc_'+str(shortname)+'/ERA5_'+str(format(dt.year, '04'))+'-'+str(format(dt.month, '02'))+'-'+str(format(dt.day, '02'))+'T'+str(format(dt.hour, '02'))+'_00_00_'+str(shortname)+'.nc'
            try:
                if os.path.isfile(ncfile):
                    logger.debug("download pressure: file already present (race) %s", ncfile)
                    logger.info("Skipped pressure download (already present): %s", ncfile)
                    return
                logger.info("Requesting pressure-level file from CDS: %s (lev=%s)", ncfile, lev)
                ERA5.c.retrieve(
                    'reanalysis-era5-pressure-levels',
                    {
                        'product_type': 'reanalysis',
                        'variable': str(name),
                        'pressure_level': str(lev),
                        'year': str(dt.year),
                        'month': str(dt.month),
                        'day': str(dt.day),
                        'valid_time': str(dt.strftime('%H:%M')),
                        'format': 'netcdf'
                    },
                    str(ncfile))
                logger.info("Saved pressure-level file: %s", ncfile)
                logger.info("Finished pressure download for %s lev=%s at %s", shortname, lev, dt)
            except Exception as e:
                logger.exception("Failed to download pressure-level %s (lev=%s) for %s: %s", name, lev, dt, e)
                logger.debug(traceback.format_exc())

        max_workers = 8
        logger.info("Starting ThreadPoolExecutor with max_workers=%d for %s lev=%s (missing=%d)", max_workers, shortname, lev, len(missing))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(download, d) for d in missing]
            for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Downloading {shortname}")):
                try:
                    future.result()
                    if (i+1) % 10 == 0:
                        logger.info("Progress: %d/%d pressure downloads completed for %s lev=%s", i+1, len(futures), shortname, lev)
                except Exception as e:
                    logger.exception("Download failed in executor: %s", e)

    ################################################################################
    # 変数
    ################################################################################
    # 時間間隔
    delta = timedelta(hours=1)

    # CDS API
    c = cdsapi.Client()
