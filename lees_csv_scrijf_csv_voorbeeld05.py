# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:29:28 2019

@author: janant
"""
import os
from pandas import DataFrame, read_csv
import pandas as pd 
import numpy as np
import global_settings as gs
import classification_preprocess as class_pre


# STEP 1 
#  VARIABLES
#****************************************************************
# the input file "MONGROEPEN_new3 .csv"  is the result of a query on the PRI DB, 
# it is a classification of all (292) unique cropnames / classnames of the campagne year 2018 
# into theoreticaly ( 82 + 4 technical* + 3 special**) MON_groups of which only (34 + 4 technical*) appear or are processed
# and the classification of all (89 theoretical) or (38 in practise) MON_groups into (8) MON_LANDCOVER_GROUPS
# note: there are 7 special technical MON_groups values with fake Gewascode and description
#  99991 NOT_ENOUGH_PIXELS** 
#  99992 NODATA**
#  99993 MON_BONEN_WIKKEN*
#  99994 DOUBT**
#  99995 MON_BOOMKWEEK*
#  99996 MON_CONTAINERS*
#  99997 MON_STAL_SER*
#****************************************************************
# read the MONGROEPEN_new3csv into a dataframe
df_mongroep= pd.read_csv(r"X:\Monitoring\Markers\PlayGround\JanAnt\InputData\MONGROEPEN_new3.csv",sep=';')

# make a dataframe with an erase of the duplicates in the MON_groep values, to have a table of unique mongroups in mon lC groups for later use
df_monlcgroep=df_mongroep.drop_duplicates(subset ='MON_groep')

# Read out the temp predictions output csv file of the classtype_to_prepare = 'MONITORING_CROPGROUPS' process , that file is 400MB 
# df_output_predictions= pd.read_csv(r"X:\Monitoring\Markers\PlayGround\JanAnt\temp\temp_output_predictions_mini05.csv",sep=';', decimal=',' ) # decimal : str, default ‘.’ e.g. use ‘,’ for European data
#df_output_predictions= pd.read_csv(r"X:\Monitoring\Markers\PlayGround\JanAnt\temp\mini5.csv",sep=',' ) # , decimal=','  decimal : str, default ‘.’ e.g. use ‘,’ for European data
# This file commes out of classification_sklearn.py 
df_output_predictions= pd.read_csv(r"X:\Monitoring\Markers\PlayGround\JanAnt\temp\temp_output_predictions.csv",sep=',' )
#temp_output_predictions.csv

# Get list from column headers , there are 38 prediction classes /some of them are technical like MON_CONTAINERS, MON_BONEN_WIKKEN , MON_BOOMKWEEK, MON_STAL_SER..
PREDICTION_COLUMN_LIST= df_output_predictions.columns.tolist() 

# STEP 2
#****************************************************************
# A. Preprocessing to aggregate the colums for summation
# 1. Making lists of all (82+7) possible values based on the mongroep input tabel(= FULL_***_LIST)
# 2  Reducing the list of values by intersecting with occuring vallues in the prediction table (***_LIST)
#  Since some classes like MON_AUBERGINES, MON_AZALEA have not enough elements they are directly set to 'UNKNOWN', 
#  so they don't occur in the prediction csv
# the sum of the 8 different lists is 86, since the prediction list cannot contain the 3 special** codes
#****************************************************************
FULL_MON_LC_ARABLE_LIST= df_mongroep[df_mongroep['MON_LC_groep'] == 'MON_LC_ARABLE']['MON_groep'].unique().tolist() # number=67
FULL_MON_LC_FABACEAE_LIST = df_mongroep[df_mongroep['MON_LC_groep'] == 'MON_LC_FABACEAE']['MON_groep'].unique().tolist() # number=9
FULL_MON_LC_FALLOW_LIST = df_mongroep[df_mongroep['MON_LC_groep'] == 'MON_LC_FALLOW']['MON_groep'].unique().tolist() # number=2
FULL_MON_LC_GRASSES_LIST = df_mongroep[df_mongroep['MON_LC_groep'] == 'MON_LC_GRASSES']['MON_groep'].unique().tolist() # number=1
FULL_MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_LIST = df_mongroep[df_mongroep['MON_LC_groep'] == 'MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS']['MON_groep'].unique().tolist() # number=1
FULL_MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_NS_LIST = df_mongroep[df_mongroep['MON_LC_groep'] == 'MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_NS']['MON_groep'].unique().tolist() # number=4
FULL_MON_LC_INELIGIBLE_LIST = df_mongroep[df_mongroep['MON_LC_groep'] == 'MON_LC_INELIGIBLE']['MON_groep'].unique().tolist() # number=1
FULL_MON_LC_UNKNOWN_LIST  = df_mongroep[df_mongroep['MON_LC_groep'] == 'MON_LC_UNKNOWN']['MON_groep'].unique().tolist() # number =1
# Reducing the FULL Lists to list with values that occur in the prediction table , sum is 38
MON_LC_ARABLE_LIST=  set(FULL_MON_LC_ARABLE_LIST ).intersection(PREDICTION_COLUMN_LIST) # number = 26
MON_LC_FABACEAE_LIST =  set(FULL_MON_LC_FABACEAE_LIST ).intersection(PREDICTION_COLUMN_LIST) # number =3
MON_LC_FALLOW_LIST=  set(FULL_MON_LC_FALLOW_LIST ).intersection(PREDICTION_COLUMN_LIST) # number =
MON_LC_GRASSES_LIST = set(FULL_MON_LC_GRASSES_LIST ).intersection(PREDICTION_COLUMN_LIST) # number =
MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_LIST = set(FULL_MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_LIST ).intersection(PREDICTION_COLUMN_LIST) # number =
MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_NS_LIST = set(FULL_MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_NS_LIST ).intersection(PREDICTION_COLUMN_LIST) # number =
MON_LC_INELIGIBLE_LIST = set(FULL_MON_LC_INELIGIBLE_LIST ).intersection(PREDICTION_COLUMN_LIST) # number =
MON_LC_UNKNOWN_LIST =set(FULL_MON_LC_UNKNOWN_LIST ).intersection(PREDICTION_COLUMN_LIST) # number =
# Extra SUM List of the SUMS of the MON_LC_GROUPS SUM for check reasons
SUM_SUM_LIST = ["ARABLE_SUM", "FABACEAE_SUM","FALLOW_SUM", "GRASSES_SUM", "IGNORE_DIFFICULT_PERMANENT_CLASS_SUM", "IGNORE_DIFFICULT_PERMANENT_CLASS_NS_SUM", "INELIGIBLE_SUM", "UNKNOWN_SUM"] # checksum

#****************************************************************
# STEP3 B
# Summing the +- 40 39? or 38 different individual prediction columns to 8 sums columns
# plus 1 sum-sum for check reasons
#****************************************************************
df_output_predictions['ARABLE_SUM'] = df_output_predictions[MON_LC_ARABLE_LIST].sum(axis=1)
df_output_predictions['FABACEAE_SUM'] = df_output_predictions[MON_LC_FABACEAE_LIST].sum(axis=1)
df_output_predictions['FALLOW_SUM'] = df_output_predictions[MON_LC_FALLOW_LIST].sum(axis=1)
df_output_predictions['GRASSES_SUM'] = df_output_predictions[MON_LC_GRASSES_LIST].sum(axis=1)
df_output_predictions['IGNORE_DIFFICULT_PERMANENT_CLASS_SUM'] = df_output_predictions[MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_LIST].sum(axis=1)
df_output_predictions['IGNORE_DIFFICULT_PERMANENT_CLASS_NS_SUM'] = df_output_predictions[MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_NS_LIST].sum(axis=1)
df_output_predictions['INELIGIBLE_SUM'] = df_output_predictions[MON_LC_INELIGIBLE_LIST].sum(axis=1)
df_output_predictions['UNKNOWN_SUM'] = df_output_predictions[MON_LC_UNKNOWN_LIST].sum(axis=1)
# Check sum list
df_output_predictions['SUM_SUM'] = df_output_predictions[SUM_SUM_LIST].sum(axis=1)

# For test reasons, check if sum_sum = 1
print (df_output_predictions.head(5))

#****************************************************************
# STEP 4 / C. MAKE SORT MATRIX of only the 8 sum columns + the CODE_OBJ + classname to find the top 3 predictions
# to be used as an input for the top 3 algorithm
#****************************************************************

df_output_predictions_sums=df_output_predictions[SUM_SUM_LIST]
df_output_predictions_fortop3=df_output_predictions[['CODE_OBJ']+['classname'] + SUM_SUM_LIST]

#df_output_predictions_sums_tmp=df_output_predictions_sums.copy()

#df_probabilities=df_output_predictions_sums.copy()
# input for top 3 calculations, with 
df_proba=df_output_predictions_fortop3

# For test reasons,
print (df_output_predictions_sums.head(5))
df_output_predictions_sums.info()
print (df_proba.head(5))
df_proba.info()


#****************************************************************
# STEP 5 //  D. find the top 3
#****************************************************************

#
def _get_top_3_prediction(df_probabilities):
    """ Returns the top 3 predictions for each parcel.

    The return value will be a dataset with the following columns:
        - global_settings.id_column: id of the parcel as in the input
        - global_settings.class_column: class of the parcel as in the input
        - global_settings.prediction_cons_column: the consolidated prediction, will be 'DOUBT'
          if the prediction had a relatively low probability.
        - global_settings.prediction_columns: prediction with the highest probability
        - pred1_prob: probability of the best prediction
        - pred2
        - pred2_prob
        - pred3
        - pred3_prob
    """

#    logger.info("get_top_3_predictions: start")
    df_probabilities_tmp = df_probabilities.copy()
    for column in df_probabilities_tmp.columns:
        if column in gs.dedicated_data_columns:
            df_probabilities_tmp.drop(column, axis=1, inplace=True)

    # Get the top 3 predictions for each row
    # First get the indeces of the top 3 predictions for each row
    # Remark: argsort sorts ascending, so we need to take:
    #     - "[:,": for all rows
    #     - ":-4": the last 3 elements of the values
    #     - ":-1]": and than reverse the order with a negative step
    top3_pred_classes_idx = np.argsort(df_probabilities_tmp.values, axis=1)[:, :-4:-1]
    # Convert the indeces to classes
    top3_pred_classes = np.take(df_probabilities_tmp.columns, top3_pred_classes_idx)
    # Get the values of the top 3 predictions
    top3_pred_values = np.sort(df_probabilities_tmp.values, axis=1)[:, :-4:-1]
    # Concatenate both
    top3_pred = np.concatenate([top3_pred_classes, top3_pred_values], axis=1)
    # Concatenate the ids, the classes and the top3 predictions
    id_class_top3 = np.concatenate([df_probabilities[[gs.id_column, gs.class_column]].values, top3_pred]
                                   , axis=1)

    # Convert to dataframe, combine with input data and write to file
    df_top3 = pd.DataFrame(id_class_top3,
                           columns=[gs.id_column, gs.class_column,
                                    gs.prediction_column, 'pred2', 'pred3',
                                    'pred1_prob', 'pred2_prob', 'pred3_prob'])

    return df_top3

#

# Calculate the top 3 predictions - naar beneden geplaatst

# STEP 6 calculate the df_output_predictions_fortop3 (514116)
df_proba_top3 = _get_top_3_prediction(df_proba)
# original was het df_top3 maar dat was verwarrend , daarom hernoemd naar df_proba_top3 , commentaar mag weg
    # Add the consolidated prediction
#**************************************************    
def calculate_consolidated_prediction(row):
        # For some reason the row['pred2_prob'] is sometimes seen as string, and so 2* gives a
        # repetition of the string value instead of a mathematic multiplication... so cast to float!
        if float(row['pred1_prob']) >= 2.0 * float(row['pred2_prob']):
            return row[gs.prediction_column]
        else:
            return 'DOUBT'
#*****************************************************
# Apply a function along an axis of the DataFrame. axis=1 => row
values = df_proba_top3.apply(calculate_consolidated_prediction, axis=1)
df_proba_top3.insert(loc=2, column=gs.prediction_cons_column, value=values)

    # Make sure all input parcels are in the output. If there was no prediction, it means that there
    # was no data available for a classification, so set prediction to NODATA
df_proba_top3.set_index(gs.id_column, inplace=True)

# eerst parcel tabel binnenhalen.. meest van onderstaande variabel zullen binnenkort overbodig zijn
i=17
year=2018
base_dir = 'x:\\Monitoring\\Markers\\PlayGround\\JanAnt'   
class_base_dir = os.path.join(base_dir, f"{year}_class_maincrops_mon")
class_dir = os.path.join(class_base_dir, f"Run_{i+1:03d}")
input_parcel_filename_noext = 'Prc_BEFL_2018_2018-08-02' 
parcel_csv = os.path.join(class_dir, f"{input_parcel_filename_noext}_parcel.csv")
input_parcel_csv=parcel_csv
df_input_parcel = pd.read_csv(input_parcel_csv, low_memory=False)
df_input_parcel.head(5) # For test reasons
df_input_parcel.set_index(gs.id_column, inplace=True) # werkt pas als bovenstaadn is ingevoegd 
#*******
country_code = 'BEFL'
base_filename = f"{country_code}{year}_bufm10_weekly"
parcel_predictions_all_csv = os.path.join(class_dir, f"{base_filename}_predict_all.csv")
#******* PAS OP DEZE STATUS VERANDERING OP df_pred DOEN ????
    # Add a column with the prediction status... and all parcels in df_proba_top3 got a prediction
df_proba_top3[gs.prediction_status] = 'OK'
df_proba_top3.loc[(df_proba_top3[gs.prediction_cons_column] == 'DOUBT'), gs.prediction_status] = 'DOUBT'
# OK tot hier
# Join the 2 table and Add NODATA where NaN in next colums

cols_to_join = df_proba_top3.columns.difference(df_input_parcel.columns)
df_pred = df_input_parcel.join(df_proba_top3[cols_to_join], how='left')
df_pred[gs.prediction_column].fillna('NODATA', inplace=True)
df_pred[gs.prediction_cons_column].fillna('NODATA', inplace=True)
df_pred[gs.prediction_status].fillna('NODATA', inplace=True)

#logger.info(f"Columns of df_pred: {df_pred.columns}")
# Parcels with too few pixels don't have a good accuracy and give many alfa errors...
df_pred.loc[(df_pred[gs.pixcount_s1s2_column] <= 10)
                 & (df_pred[gs.prediction_status] != 'NODATA')
                 & (df_pred[gs.prediction_status] != 'DOUBT'),
                [gs.prediction_cons_column, gs.prediction_status]] = 'NOT_ENOUGH_PIXELS'

df_pred.loc[df_pred[gs.class_column] == 'UNKNOWN', [gs.prediction_status]] = 'UNKNOWN'
## PAS op ook de "MON_ONBEKEND_MET_KLASSIFICATIE" meenemen
df_pred.loc[df_pred[gs.class_column] == 'MON_ONBEKEND_MET_KLASSIFICATIE', [gs.prediction_status]] = 'MON_ONBEKEND_MET_KLASSIFICATIE'
df_pred.loc[df_pred[gs.class_column].str.startswith('IGNORE_'), [gs.prediction_status]] = df_pred[gs.class_column]
## PAS op ook de "MON_MOEILIJK_ZONDER_KLASSIFICATIE" 
df_pred.loc[df_pred[gs.class_column].str.startswith('MON_MOEILIJK_'), [gs.prediction_status]] = df_pred[gs.class_column]

#logger.info("Write final prediction data to file")
# VERDER GAAN MET DEZE  ipv de verkeerde lager!!!!!!!!!!!!!!!!!!!!!!

df_pred.to_csv(r'x:\Monitoring\Markers\PlayGround\JanAnt\2018_class_landcover_via_cropgroup_mon_jan\df_pred0.csv')
df_pred_to_csv = (r'x:\Monitoring\Markers\PlayGround\JanAnt\2018_class_landcover_via_cropgroup_mon_jan\df_pred4.csv')
# output_predictions_csv=output_predictions_postpr_test_csv

# Herclasseren van classname in df_pred , voor het rapport!
# df_pred_reclassed1.replace(list(MON_LC_ARABLE_LIST), 'ARABLE_SUM')

df_pred_reclassed= df_pred
df_pred_reclassed

df_pred_reclassed.loc[df_pred_reclassed["classname"].isin(list(MON_LC_ARABLE_LIST)), 'classname'] = 'ARABLE_SUM'
df_pred_reclassed.loc[df_pred_reclassed["classname"].isin(list(MON_LC_FABACEAE_LIST)), 'classname'] = 'FABACEAE_SUM'
df_pred_reclassed.loc[df_pred_reclassed["classname"].isin(list(MON_LC_FALLOW_LIST)), 'classname'] = 'FALLOW_SUM'
df_pred_reclassed.loc[df_pred_reclassed["classname"].isin(list(MON_LC_GRASSES_LIST)), 'classname'] = 'GRASSES_SUM'
df_pred_reclassed.loc[df_pred_reclassed["classname"].isin(list(MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_LIST)), 'classname'] = 'IGNORE_DIFFICULT_PERMANENT_CLASS_SUM'
df_pred_reclassed.loc[df_pred_reclassed["classname"].isin(list(MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_NS_LIST)), 'classname'] = 'IGNORE_DIFFICULT_PERMANENT_CLASS_NS_SUM'
df_pred_reclassed.loc[df_pred_reclassed["classname"].isin(list(MON_LC_INELIGIBLE_LIST)), 'classname'] = 'INELIGIBLE_SUM'
df_pred_reclassed.loc[df_pred_reclassed["classname"].isin(list(MON_LC_UNKNOWN_LIST)), 'classname'] = 'INELIGIBLE_SUM'

df_pred_reclassed.to_csv(r'x:\Monitoring\Markers\PlayGround\JanAnt\2018_class_landcover_via_cropgroup_mon_jan\df_pred5.csv')

df_pred_reclassed_to_csv = (r'x:\Monitoring\Markers\PlayGround\JanAnt\2018_class_landcover_via_cropgroup_mon_jan\df_pred5.csv')

'''
#*********************************************
#OVERBODIG  
#STEP 7 Prepareren van de TOP3 in juist formaat? 
#df_proba_top3.to_csv(output_predictions_csv, float_format='%.10f', encoding='utf-8')
# df_proba_top3.to_csv(r'x:\Monitoring\Markers\PlayGround\JanAnt\2018_class_landcover_via_cropgroup_mon_jan\top3_test.csv')

#pred_all= pd.read_csv(r"X:\Monitoring\Markers\playground\janant\2018_class_landcover_via_cropgroup_mon_jan\BEFL2018_bufm10_weekly_predict_all.csv",sep=',')
#pred_all_index=pred_all.set_index ('CODE_OBJ')

# adds "_sum" to the columns names that start with "pred" -to distinguish after joining with pred
# df_proba_top3_suffix = df_proba_top3_index.add_suffix('_sum') is ook goed hoor ;)
#df_proba_top3_suffix =df_proba_top3[df_proba_top3.columns[pd.Series(df_proba_top3.columns).str.startswith('pred')]].add_suffix('_sum') # enkel voor selectie start with 'pred'

# BAD !! join top3 with predictions all

#df_joined=  pred_all_index.join(df_proba_top3_suffix.set_index('CODE_OBJ'), on='CODE_OBJ')
#df_joined= pred_all_index.join(df_proba_top3_suffix, lsuffix='_left', rsuffix='_right')
#df_joined.to_csv(R'x:\Monitoring\Markers\PlayGround\JanAnt\2018_class_landcover_via_cropgroup_mon_jan\df_joined.csv')

#pred_via = df_joined[[]]
'''



# LAST STEP 6: Report on the test accuracy, incl. ground truth
#-------------------------------------------------------------
# EXTRA INPUTS FOR TEST

import classification
import classification_reporting as class_report
input_dir = os.path.join(base_dir, 'InputData')     
input_groundtruth_csv = os.path.join(input_dir, "Prc_BEFL_2018_groundtruth.csv")
input_parcel_filetype = country_code
imagedata_dir = os.path.join(base_dir, 'Timeseries_data')      # General data download dir
parcel_pixcount_csv = os.path.join(imagedata_dir, f"{base_filename}_pixcount.csv")
#classtype_to_prepare = 'MONITORING_LANDCOVER_VIA_CROPGROUPS'
classtype_to_prepare = 'MONITORING_CROPGROUPS'

parcel_predictions_test_csv = os.path.join(class_dir, f"{base_filename}_predict_test.csv") # 'x:\\Monitoring\\Markers\\PlayGround\\JanAnt\\2018_class_landcover_mon\\Run_012\\BEFL2018_bufm10_weekly_predict_test.csv'
parcel_predictions_all_csv = os.path.join(class_dir, f"{base_filename}_predict_all.csv") # 'x:\\Monitoring\\Markers\\PlayGround\\JanAnt\\2018_class_landcover_mon\\Run_012\\BEFL2018_bufm10_weekly_predict_all.csv'

#


# Preprocess the ground truth data
groundtruth_csv = None
if input_groundtruth_csv is not None:
    input_gt_noext, input_gt_ext = os.path.splitext(input_groundtruth_csv) # Split the pathname path into a pair (root, ext)
    groundtruth_csv = os.path.join(class_dir, f"{input_gt_noext}_classes{input_gt_ext}")
    class_pre.prepare_input(input_parcel_filepath=input_groundtruth_csv,
                            input_filetype=input_parcel_filetype,
                            input_parcel_pixcount_csv=parcel_pixcount_csv,
                            output_parcel_filepath=groundtruth_csv,
                            input_classtype_to_prepare=f"{classtype_to_prepare}_GROUNDTRUTH")

# Print full reporting on the accuracy
report_txt = f"{parcel_predictions_test_csv}_accuracy_report_jan01c.txt" # x:\\Monitoring\\Markers\\PlayGround\\JanAnt\\2018_class_landcover_mon\\Run_012\\BEFL2018_bufm10_weekly_predict_test.csv_accuracy_report.txt'
#class_report.write_full_report(parcel_predictions_csv=parcel_predictions_postpr_test_csv,
## Writes output 9 BEFL2018_bufm10_weekly_predict_test.csv_accuracy_report.txtgroundtruth_pred_quality_details.csv
## Writes output 10 BEFL2018_bufm10_weekly_predict_test.csv_accuracy_report.txt
## Writes output 11 BEFL2018_bufm10_weekly_predict_test.csv_accuracy_report.html
## class_report.write_full_report(parcel_predictions_csv=parcel_predictions_test_csv,
class_report.write_full_report(parcel_predictions_csv=parcel_predictions_test_csv,
                               output_report_txt=report_txt,
                               parcel_ground_truth_csv=groundtruth_csv)

# STEP 7: Report on the full accuracy, incl. ground truth
#-------------------------------------------------------------
# Print full reporting on the accuracy
report_txt = f"{parcel_predictions_all_csv}_accuracy_report_jan05.txt"
#class_report.write_full_report(parcel_predictions_csv=parcel_predictions_postpr_all_csv,
#class_report.write_full_report(parcel_predictions_csv=parcel_predictions_all_csv,
# class_report.write_full_report(parcel_predictions_csv=df_pred_to_csv,
class_report.write_full_report(parcel_predictions_csv=df_pred_reclassed_to_csv,
                               output_report_txt=report_txt,
                               parcel_ground_truth_csv=groundtruth_csv)




'''
logging.shutdown()
'''








# If the script is run directly...
#if __name__ == "__main__":
#    logger.critical('Not implemented exception!')
#    raise Exception('Not implemented')