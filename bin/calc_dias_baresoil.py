# -*- coding: utf-8 -*-
"""
Create timeseries data vegegation/bare soil per image on DIAS.
"""

import os
import logging
import pandas as pd

import sys
[sys.path.append(i) for i in ['.', '..']]

from cropclassification.preprocess import timeseries_util as ts_util
from cropclassification.helpers import pandas_helper as pdh
from cropclassification.helpers import config_helper as conf


"""
from datetime import datetime
import glob
import shutil

from cropclassification.helpers import config_helper as conf
from cropclassification.helpers import log_helper
from cropclassification.preprocess import timeseries_calc_dias_onda_per_image as calc_ts
"""

def Summarize_Coverage_Bare():
    
    # Get a logger...
    logger = logging.getLogger(__name__)

    # Init    
    input_dir = r'X:\Monitoring\Markers\playground\market\_timeseries_per_image\Prc_BEFL_2019_2019-08-14_bufm5'
    input_ext = '.sqlite'
   
    # Create Dataframe of all SCL files with their info
    logger.debug('Create Dataframe with all files and their properties')
    
    filelist = []
    for filename in os.listdir(input_dir): 
        if filename.endswith(input_ext):   
            
            # Get SLC files
            file_info = ts_util.get_file_info(os.path.join(input_dir, filename)) 
            if file_info["band"] == 'SCL-20m':
                # print(file_info["filepath"])
                filelist.append(file_info) 
                

            # todo marina: nog filteren op start en einddatum :tch niet dit wordt doorgegeven vanuit calc_dias

    SCL_imagefiles_df = pd.DataFrame(filelist)
    
    # Write to file                                                              #temp - mag nadien weg
    output_dir = r'X:\Monitoring\Markers\playground\market\output\baresoil'
    output_ext = '.csv'
    output_filename = f"Lijst_alle_SCLbeelden{output_ext}" 
    output_filepath = os.path.join(output_dir, output_filename)
    pdh.to_file(SCL_imagefiles_df, output_filepath)

    SCL_imagefiles_df = SCL_imagefiles_df.sort_values('filename', ascending=False)

    # Init
    Columns_to_use_dict = {'UID':[], 'vegetation':[], 'not_vegetated':[]}
    period_band_data_df = None

    # Read the SCL files and get the necessary info
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
        columns = [column for column in Columns_to_use_dict]
        image_data_df = pdh.read_file(imagedata_filepath, columns=columns)
        image_data_df['Date'] = date_fromfile


        image_data_df.set_index('UID', inplace=True)
        # image_data_df.index.name = 'UID'

        # Remove rows with nan values
        nb_before_dropna = len(image_data_df.index)
        image_data_df.dropna(inplace=True)
        nb_after_dropna = len(image_data_df.index)
        if nb_after_dropna != nb_before_dropna:
            logger.warning(f"Before dropna: {nb_before_dropna}, after: {nb_after_dropna} for file {imagedata_filepath}")
        if nb_after_dropna == 0:
            continue

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
            period_band_data_df = pd.concat([period_band_data_df, image_data_df], axis=0) #, sort=False)  #marina: check: wat doet sort?
            # Apparently concat removes the index name in some situations
            period_band_data_df.index.name = 'UID'

        
        # if j == 5:
        #     break

    # Write to file                                                                 #temp - mag nadien weg
    output_dir = r'X:\Monitoring\Markers\playground\market\output\baresoil'
    output_ext = '.csv'
    output_filename = f"Lijst_percelen_Datum{output_ext}" #todo : naamgeving bekijken
    output_filepath = os.path.join(output_dir, output_filename)
    pdh.to_file(period_band_data_df, output_filepath)

    print('end')
    

if __name__ == '__main__':
    Summarize_Coverage_Bare()
