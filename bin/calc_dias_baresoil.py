# -*- coding: utf-8 -*-
"""
Create timeseries data vegegation/bare soil per image on DIAS.
"""

import datetime
import logging
import os
import pandas as pd

import sys
[sys.path.append(i) for i in ['.', '..']]

from cropclassification.preprocess import timeseries_util as ts_util
from cropclassification.helpers import pandas_helper as pdh
from cropclassification.helpers import config_helper as conf
from cropclassification.helpers import log_helper

"""
import glob
import shutil

from cropclassification.preprocess import timeseries_calc_dias_onda_per_image as calc_ts
"""

def summarize_coverage_bare():

    year = 2018

    # We moeten eigenlijk weten voor de percelen aardappelen, silomaïs en late ajuinen, wanneer de 
    # hoofdteelt geoogst is voor 15/10 en er dus in principe tijdig een nateelt ingezaaid kon worden.
    # Dus enkel voor de percelen met een nateelt.

    # Determine the config files to load depending on the marker_type
    config_filepaths = ["config/general.ini",
                        "config/local_overrule.ini"]
    # Read the configuration files
    conf.read_config(config_filepaths, year=year)

    # Init logging
    base_log_dir = conf.dirs['log_dir']
    log_dir = f"{base_log_dir}{os.sep}calc_dias_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"

    global logger
    logger = log_helper.main_log_init(log_dir, __name__)
    logger.info(f"Config used: \n{conf.pformat_config()}")

    # Init    
    temp_dir = conf.dirs['temp_dir']
    input_ext = '.sqlite'
    output_ext = '.sqlite'

    input_preprocessed_dir = conf.dirs['input_preprocessed_dir']
    parcels_layer = 'Prc_BEFL_2018_2019-06-14'
    parcels_prep_layer = 'Prc_BEFL_2018_2019-06-14_bufm5'
    parcels_filepath = os.path.join(input_preprocessed_dir, parcels_layer + ".sqlite")

    timeseries_per_image_basedir = conf.dirs['timeseries_per_image_dir']
    timeseries_per_image_dir = os.path.join(timeseries_per_image_basedir, parcels_prep_layer)

    start_date = datetime.datetime(year, 5, 1)
    stop_date = datetime.datetime(year+1, 3, 1)

    # Filter list
    uid_list = None
    '''
    uid_list = ['0000280665CFE2BD00000002', '00002806611F20B900000004', '00002806657B0F7700000001',
                '000028065F0DB12B00000001', '000028065D96066C00000002', '000028064C8AFF5000000001',
                '0000280649E98CF800000004', '0000280662EB4AAD00000001', '000028064D2E604000000003', 
                '00002806652A721800000001', '0000280648485D8000000001', '000028064ACBC92D00000001', 
                '000028064848585F00000002', '000028065BE959B400000008', '0000280664236B4C00000005', 
                '000028065EC89C8100000002', '000028065B7783DA00000002', '000028066205F4B900000003', 
                '000028065BC7F95C00000001', '000028065EBB7AE400000001', '0000280663A21EAC00000001', 
                '0000280662790CA300000008', '000028065CE8C3E300000001']
    '''
    # Filter dataframe
    parcels_df = pdh.read_file(parcels_filepath)

    # Only parcels with GWSCOD_N not null
    parcels_df = parcels_df.loc[parcels_df['GWSCOD_N'].notnull()]
    parcels_df.set_index('UID', inplace=True)

    # Create Dataframe of all SCL files with their info
    logger.debug('Create Dataframe with all files and their properties')
    
    filelist = []
    for filename in os.listdir(timeseries_per_image_dir): 
        if filename.endswith(input_ext):   
            
            # Only SCL files + filter on date
            file_info = ts_util.get_file_info(os.path.join(timeseries_per_image_dir, filename)) 
            if(file_info['band'] == 'SCL-20m'
               and file_info['date'] > start_date
               and file_info['date'] < stop_date):
                filelist.append(file_info) 
    
    if len(filelist) == 0:
        logger.warning(f"No valid SCL data files found for start_date: {start_date}, stop_date: {stop_date} in dir {timeseries_per_image_dir}")
        return
    SCL_imagefiles_df = pd.DataFrame(filelist)
    
    # Write to file                                                              #temp - mag nadien weg
    output_dir = os.path.join(temp_dir,  'output_baresoil')
    output_filename = f"Lijst_alle_SCLbeelden{output_ext}" 
    output_filepath = os.path.join(output_dir, output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pdh.to_file(SCL_imagefiles_df, output_filepath)

    SCL_imagefiles_df = SCL_imagefiles_df.sort_values('filename', ascending=False)

    # Init
    columns_to_use_dict = {'UID':[], 'vegetation':[], 'not_vegetated':[]}
    period_band_data_df = None

    # Read the SCL files and get the necessary info
    nb_files = len(SCL_imagefiles_df.index)
    for j, imagedata_filepath in enumerate(SCL_imagefiles_df.filepath.tolist()):  

        # If file has filesize == 0, skip
        if os.path.getsize(imagedata_filepath) == 0:
            continue
            
        # Get the date from the filename
        filename = os.path.split(imagedata_filepath)[1]
        datetime_fromfile = filename.split('_')[8]
        date_fromfile = datetime_fromfile.split('T')[0]

        # Read the file 
        # image_data_df = pdh.read_file(imagedata_filepath)
        
        # Keep only the necessary columns 
        columns_to_read = [column for column in columns_to_use_dict]
        columns_to_read.append('nb_bad_pixels')
        image_data_df = pdh.read_file(imagedata_filepath) #, columns=columns_to_read)
        image_data_df.set_index('UID', inplace=True)
        
        # Only parcels without bad pixels
        image_data_df = image_data_df.loc[image_data_df['nb_bad_pixels'] == 0]

        # Only parcels with > 0 nb_pixels_total
        if 'nb_pixels_total' not in image_data_df.columns:
            nb_pixels_total_columns = list(image_data_df.columns)
            nb_pixels_total_columns.remove('nb_bad_pixels')          
            image_data_df['nb_pixels_total'] = image_data_df[nb_pixels_total_columns].sum(axis=1)
        image_data_df = image_data_df.loc[image_data_df['nb_pixels_total'] > 0]

        # If list of UID provided, filter
        if uid_list is not None:
            image_data_df = image_data_df.loc[image_data_df.index.isin(uid_list)]

        # If dataframe with UID provided, filter using that
        if parcels_df is not None:
            #image_data_df = image_data_df.loc[image_data_df.index.isin(parcels_df.index)]
            parcel_columns_to_keep = ['GWSCOD_H', 'GWSCOD_N']
            image_data_df = image_data_df.join(parcels_df[parcel_columns_to_keep], how='inner')

        # Remove rows with nan values
        nb_before_dropna = len(image_data_df.index)
        image_data_df.dropna(inplace=True)
        nb_after_dropna = len(image_data_df.index)
        if nb_after_dropna != nb_before_dropna:
            logger.warning(f"Before dropna: {nb_before_dropna}, after: {nb_after_dropna} for file {imagedata_filepath}")
        if nb_after_dropna == 0:
            continue

        # Add date column
        image_data_df['date'] = date_fromfile

        # Rename columns so column names stay unique
        '''
        for Used_Column in Columns_to_use_dict:
            if Used_Column == 'UID':
                continue

            new_column_name = Used_Column + str(date_fromfile)
            image_data_df.rename(columns={Used_Column: new_column_name},
                                 inplace=True)
            #image_data_df[new_column_name] = image_data_df[new_column_name].astype(float) #?
            Columns_to_use_dict[Used_Column].append(new_column_name)
        '''
        
        # Create 1 dataframe for all files - one row for each UID - using concat (UID = index)
        if period_band_data_df is None:
            period_band_data_df = image_data_df  
            #period_band_data_df.set_index('UID', inplace=True)              
            period_band_data_df.index.name = 'UID'
        else:
            period_band_data_df = pd.concat([period_band_data_df, image_data_df], axis=0, sort=False)  #marina: check: wat doet sort?
            # Apparently concat removes the index name in some situations
            period_band_data_df.index.name = 'UID'
        
        # if j == 5:
        #     break
        logger.info(f"Processed {j} files of {nb_files}")

    # Often there are 2 images containing the parcel on the same date: one with 
    # many pixels, one with few: only keep the one with most pixels
    period_band_data_df = period_band_data_df.loc[
            period_band_data_df.groupby(['UID', 'date'])['nb_pixels_total'].idxmax()]

    # Write to file                                                                 #temp - mag nadien weg
    output_filename = f"Lijst_percelen_Datum_v3{output_ext}" #todo : naamgeving bekijken
    output_filepath = os.path.join(output_dir, output_filename)
    pdh.to_file(period_band_data_df, output_filepath)   

if __name__ == '__main__':
    summarize_coverage_bare()
