# -*- coding: utf-8 -*-
"""
Calaculate the timeseries data per image on DIAS.
"""

import datetime as dt
import glob
import json
import os
import shutil
import sys
[sys.path.append(i) for i in ['.', '..']]

from catalogclient import catalog  # to interrogate the terrascope catalog
    
from cropclassification.helpers import config_helper as conf
from cropclassification.helpers import geofile as geofile
from cropclassification.helpers import log_helper
from cropclassification.preprocess import timeseries_calc_dias_onda_per_image as calc_ts

def get_relevant_images_terrascope(
        roi_bounds: (),
        satellitetype: str,
        startdate: dt.date,
        enddate: dt.date):

    '''
    from pprint import pprint
    pprint(vars(catalog))
    pprint(vars(catalog.EOProduct))
    '''

    logger.info(f"Get {satellitetype} images in {roi_bounds} from {startdate} till {enddate}") 
    cat = catalog.Catalog()
    
    # there are many products in the list, so we want to select just the ones related to Sentinel. These have a 'CGS' prefix
    # there are also PROBA-V data, with a 'PROBAV' prefix
    # there are also SPOT VEGETATION data, with a 'SPOTVEGETATION' prefix
    producttypes = cat.get_producttypes()
    if satellitetype == 'S1_GRD':
        producttypes_toget = list(filter(lambda f: 'CGS_S1_GRD_SIGMA0_L1' in f, producttypes))
    elif satellitetype == 'S2':
        producttypes_toget = list(filter(lambda f: 'CGS_S2_CCC' in f, producttypes))
    else:
        raise Exception(f"Satellite type {satellitetype} is not supported.")

    products = None
    for producttype in producttypes_toget:
        products_found = cat.get_products(
                producttype,
                fileformat='GEOTIFF',
                startdate=startdate,
                enddate=enddate,
                min_lon=roi_bounds[0],
                max_lon=roi_bounds[1],
                min_lat=roi_bounds[2],
                max_lat=roi_bounds[3])
        if products is None:
            products = products_found
        else:
            products.append(products_found)

    print(str(len(products)) + " products found")
    for product in products:
        for b in product.bands():
            logger.info(product.file(b))

    '''
    for p in products:
        # there might be multiple bands in a single product
        # str(p) contains the datetime of the sensing
        print(str(p)[-19:-9]) #date
        print("----------------------------------------------")
        for b in p.bands(): #there are two bands - VV and VH, but they obviously point to the same datafile
            files = os.listdir(p.file(b)[5:] + "/")
            for f in files:  # there are 3 files per folder: *.xlm and *.info contain metadata
                # the SAR data itself in in the *.zip file (SNAP opens the ZIP file, no
                # need to unpack it)
                print (b + " " +p.file(b)[5:] + "/" + f)
            print()
    '''

def get_relevant_images_onda(
        features_path: str,
        satellite_type: str,
        startdate,
        enddate):
    return

def main():

    # Determine the config files to load depending on the marker_type
    config_filepaths = ["../config/general.ini",
                        "../config/local_overrule_linux.ini"]

    test = False

    # Specify the date range:

    years = [2019]
    month_start = 3
    month_stop = 8
    for year in years:

        # Read the configuration files
        conf.read_config(config_filepaths, year=year)

        # Get the general output dir
        input_preprocessed_dir = conf.dirs['input_preprocessed_dir']
        timeseries_per_image_dir = conf.dirs['timeseries_per_image_dir']

        # Init logging
        if not test:
            base_log_dir = conf.dirs['log_dir']
        else:
            base_log_dir = conf.dirs['log_dir'] + '_test'
        log_dir = f"{base_log_dir}{os.sep}calc_dias_{dt.datetime.now():%Y-%m-%d_%H-%M-%S}"

        # Clean test log dir if it exist
        if test and os.path.exists(base_log_dir):
            shutil.rmtree(base_log_dir)

        global logger
        logger = log_helper.main_log_init(log_dir, __name__)
        logger.info(f"Config used: \n{conf.pformat_config()}")

        if test:         
            logger.info(f"As we are testing, clean all test logging and use new log_dir: {log_dir}")

        # Write the consolidated config as ini file again to the run dir
        config_used_filepath = os.path.join(log_dir, 'config_used.ini')
        with open(config_used_filepath, 'w') as config_used_file:
            conf.config.write(config_used_file)

        # Input features file depends on the year
        if year == 2017:
            input_features_filename = "Prc_BEFL_2017_2019-06-14_bufm5.shp"
        elif year == 2018:
            input_features_filename = "Prc_BEFL_2018_2019-06-14_bufm5.shp"
        elif year == 2019:
            #input_features_filename = "Prc_BEFL_2019_2019-06-25_bufm5.shp"
            input_features_filename = "Prc_BEFL_2019_2019-08-14_bufm5.shp"
        else:
            raise Exception(f"Not a valid year: {year}")
        input_features_path = os.path.join(input_preprocessed_dir, input_features_filename)
        
        # Get the bounds of the geofile
        input_features_bounds_path = input_features_path + '_bounds.json'
        if os.path.exists(input_features_bounds_path):
            with open(input_features_bounds_path) as f:
                roi_bounds = json.loads(f.read())
        else:
            # Calculate the bounds
            roi_bounds, roi_crs = geofile.get_totalbounds(input_features_path)
            if roi_crs['init'] != 'epsg:4326':
                import pyproj
                roi_x, roi_y = pyproj.transform(roi_crs, pyproj.Proj(init='epsg:4326'), 
                        x=[roi_bounds[0], roi_bounds[2]], 
                        y=[roi_bounds[1], roi_bounds[3]])
                roi_bounds = (roi_x[0], roi_y[0], roi_x[1], roi_y[1])
                logger.info(f"roi needs to be in epsg:4326, result: {roi_bounds}")
            # Write to file
            with open(input_features_bounds_path, "w") as f:
                f.write(json.dumps(roi_bounds))

        # Init output dir 
        if not test:
            output_basedir = timeseries_per_image_dir
        else:
            output_basedir = timeseries_per_image_dir + '_test'
            logger.info(f"As we are testing, use test output basedir: {output_basedir}")
        input_features_filename_noext = os.path.splitext(input_features_filename)[0]
        output_dir = os.path.join(output_basedir, input_features_filename_noext)
        if test:
            if os.path.exists(output_dir):
                logger.info(f"As we are only testing, clean the output dir: {output_dir}")
                # By adding a / at the end, only the contents are recursively deleted
                shutil.rmtree(output_dir + os.sep)
    
        # Temp dir + clean contents from it.
        temp_dir = conf.dirs['temp_dir'] + os.sep + 'calc_dias'
        logger.info(f"Clean the temp dir {temp_dir}")
        if os.path.exists(temp_dir):
            # By adding a / at the end, only the contents are recursively deleted
            shutil.rmtree(temp_dir + os.sep)
            
        """
        # TEST to extract exact footprint from S1 image...
        filepath = "/mnt/NAS3/CARD/FLANDERS/S1A/L1TC/2017/01/01/S1A_IW_GRDH_1SDV_20170101T055005_20170101T055030_014634_017CB9_Orb_RBN_RTN_Cal_TC.CARD/S1A_IW_GRDH_1SDV_20170101T055005_20170101T055030_014634_017CB9_Orb_RBN_RTN_Cal_TC.data/Gamma0_VH.img"
        image = rasterio.open(filepath)
        geoms = list(rasterio.features.dataset_features(src=image, as_mask=True, precision=5))
        footprint = gpd.GeoDataFrame.from_features(geoms)        
        logger.info(footprint)
        footprint = footprint.simplify(0.00001)        
        logger.info(footprint)
        logger.info("Ready")
        # Start calculation
        """
    
        startdate = dt.date(year, month_start, 1)
        enddate = dt.date(year, month_stop, 31)

        get_relevant_images_terrascope(
                roi_bounds=roi_bounds,
                satellitetype='S1_GRD',
                startdate=startdate,
                enddate=enddate)

        """
        ##### Process S1 GRD images #####
        input_image_filepaths = []
        for i in range(month_start, month_stop+1):
            # ONDA search string
            #input_image_searchstr = f"/mnt/NAS3/CARD/FLANDERS/S1*/L1TC/{year}/{i:02d}/*/*.CARD"
            # VITO search string
            input_image_searchstr = f"/mnt/NAS3/CARD/FLANDERS/S1*/L1TC/{year}/{i:02d}/*/*.CARD"

            input_image_filepaths.extend(glob.glob(input_image_searchstr))
        logger.info(f"Found {len(input_image_filepaths)} S1 GRD images to process")

        if test:
            # Take only the x first images found while testing
            input_image_filepaths = input_image_filepaths[:10]
            logger.info(f"As we are only testing, process only {len(input_image_filepaths)} test images")

        calc_ts.calc_stats_per_image(
                features_filepath=input_features_filepath,
                id_column=conf.columns['id'],
                image_paths=input_image_filepaths,
                bands=['VV', 'VH'],
                output_dir=output_dir,
                temp_dir=temp_dir,
                log_dir=log_dir)

        ##### Process S2 images #####
        input_image_filepaths = []
        for i in range(month_start, month_stop+1):
            # ONDA search string
            #input_image_searchstr = f"/mnt/NAS3/CARD/FLANDERS/S2*/L2A/{year}/{i:02d}/*/*.SAFE"
            # VITO search string
            input_image_searchstr = f"/data/sentinel_data/sentinel2/raw/l2a_esa/{year}/{i:02d}/*/*.SAFE"

            input_image_filepaths.extend(glob.glob(input_image_searchstr))    
        logger.info(f"Found {len(input_image_filepaths)} S2 images to process")

        if test:
            # Take only the x first images found while testing
            input_image_filepaths = input_image_filepaths[:10]
            logger.info(f"As we are only testing, process only {len(input_image_filepaths)} test images")

        # TODO: refactor underlying code so the SCL band is used regardless of it being passed here
        max_cloudcover_pct = conf.timeseries.getfloat('max_cloudcover_pct')
        calc_ts.calc_stats_per_image(
                features_filepath=input_features_filepath,
                id_column=conf.columns['id'],
                image_paths=input_image_filepaths,
                bands=['B02-10m', 'B03-10m', 'B04-10m', 'B08-10m', 'SCL-20m'],
                output_dir=output_dir,
                temp_dir=temp_dir,
                log_dir=log_dir,
                max_cloudcover_pct=max_cloudcover_pct)

        ##### Process S1 Coherence images #####   
        input_image_filepaths = []
        for i in range(month_start, month_stop+1):
            # ONDA search string
            #input_image_searchstr = f"/mnt/NAS3/CARD/FLANDERS/S1*/L1CO/{year}/{i:02d}/*/*.CARD"
            # VITO search string
            input_image_searchstr = f"/mnt/NAS3/CARD/FLANDERS/S1*/L1CO/{year}/{i:02d}/*/*.CARD"

            input_image_filepaths.extend(glob.glob(input_image_searchstr))  
        logger.info(f"Found {len(input_image_filepaths)} S1 Coherence images to process")

        if test:
            # Take only the x first images found while testing
            input_image_filepaths = input_image_filepaths[:10]
            logger.info(f"As we are only testing, process only {len(input_image_filepaths)} test images")

        calc_ts.calc_stats_per_image(features_filepath=input_features_filepath,
                id_column=conf.columns['id'],
                image_paths=input_image_filepaths,
                bands=['VV', 'VH'],
                output_dir=output_dir,
                temp_dir=temp_dir,
                log_dir=log_dir)
        """

if __name__ == '__main__':
    main()
