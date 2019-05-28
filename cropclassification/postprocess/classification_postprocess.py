# -*- coding: utf-8 -*-
"""
Module with postprocessing functions on classification results.
"""

import logging
import os

import numpy as np
import pandas as pd

import cropclassification.helpers.config_helper as conf
import cropclassification.helpers.pandas_helper as pdh

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------
# def reclassify(input_predictions_proba_filepath: str,
#                reclassify_type: str,
#                output_predictions_proba_filepath: str):
    
#     return

def reclassify(input_parcel_filepath: str, 
               input_predictions_proba_filepath: str,
               output_parcel_filepath: str,
               output_predictions_proba_filepath: str):
    """
    Commentaar...
    """
    # first read classification csv
    reclassify_dest_class_columnname = 'MON_LC_GROUP'
    reclassify_src_class_columnname = 'MON_CROPGROUP'
    classname_before_reclassify = "classname_before_reclassify"
    input_dir=conf.dirs['input_dir']
    input_classes_filepath = os.path.join(input_dir, "refe_mon_cropgroups_landcover_2018.csv")
    df_refe_mon_cropgroups_landcover_2018 = pd.read_csv(input_classes_filepath, sep=',', encoding='cp1252')
    # read input parcel file
    input_parcel_df = pdh.read_file(input_parcel_filepath)
    input_parcel_df[classname_before_reclassify] = input_parcel_df[conf.columns['class']]

    reclassify_unique = df_refe_mon_cropgroups_landcover_2018[[reclassify_dest_class_columnname, reclassify_src_class_columnname]].drop_duplicates()

    # Set the indexes to join    
    if reclassify_unique.index.name != reclassify_src_class_columnname:
        reclassify_unique.set_index(reclassify_src_class_columnname, inplace=True)
    if input_parcel_df.index.name != conf.columns['class']:
        input_parcel_df.reset_index(inplace=True)
        input_parcel_df.set_index(conf.columns['class'], inplace=True)
    # Reclass input parcelfile + write to output

    

    input_parcelreclassified_df = input_parcel_df.join(reclassify_unique[reclassify_dest_class_columnname], how='left')     
     
    input_parcelreclassified_df.rename(columns={reclassify_dest_class_columnname: conf.columns['class']}, inplace=True)
    input_parcelreclassified_df.reset_index(drop=True, inplace=True)
    
    pdh.to_file(input_parcelreclassified_df, output_parcel_filepath, index=False)
    df_refe_mon_cropgroups_landcover_2018.reset_index(inplace=True)
    
    # then read the prediction input file
    df_parcel_predictions_proba = pdh.read_file(input_predictions_proba_filepath)
    
    """
    # Get list of groups to reclassify to
    reclassify_src_class_columnname = 'MON_LC_GROUP'
    reclassify_dest_class_columnname = 'MON_CROPGROUP'
    reclassify_classes_list = df_refe_mon_cropgroups_landcover_2018[reclassify_src_class_columnname].unique().tolist()

    prediction_proba_column_list = df_parcel_predictions_proba.columns.tolist() 
    for reclassify_class in reclassify_classes_list:
        source_classes = df_refe_mon_cropgroups_landcover_2018[df_refe_mon_cropgroups_landcover_2018[reclassify_src_class_columnname] == reclassify_class][reclassify_dest_class_columnname].unique().tolist()
        source_classes_intersect = set(source_classes).intersection(prediction_proba_column_list)
        df_parcel_predictions_proba[reclassify_class] = df_parcel_predictions_proba[source_classes_intersect].sum(axis=1)

    df_output_predictions_fortop3 = df_parcel_predictions_proba[['UID']+['classname'] + reclassify_classes_list]

    # Write the output file
    pdh.to_file(df_output_predictions_fortop3, output_predictions_proba_filepath)
    """

    #"""
    # STEP 2_1 Making lists of all possible MON_CROPGROUP values (82) per MON_LC_GROUP based on the input tabel(= ALL_***_LIST)
    ALL_MON_CROPGROUPS_IN_MON_LC_ARABLE_LIST= df_refe_mon_cropgroups_landcover_2018[df_refe_mon_cropgroups_landcover_2018['MON_LC_GROUP'] == 'MON_LC_ARABLE']['MON_CROPGROUP'].unique().tolist() # number=67
    ALL_MON_CROPGROUPS_IN_MON_LC_FABACEAE_LIST = df_refe_mon_cropgroups_landcover_2018[df_refe_mon_cropgroups_landcover_2018['MON_LC_GROUP'] == 'MON_LC_FABACEAE']['MON_CROPGROUP'].unique().tolist() # number=8! not 9  NOT MON_BONEN_WIKKEN !!!
    ALL_MON_CROPGROUPS_IN_MON_LC_FALLOW_LIST = df_refe_mon_cropgroups_landcover_2018[df_refe_mon_cropgroups_landcover_2018['MON_LC_GROUP'] == 'MON_LC_FALLOW']['MON_CROPGROUP'].unique().tolist() # number=2
    ALL_MON_CROPGROUPS_IN_MON_LC_GRASSES_LIST = df_refe_mon_cropgroups_landcover_2018[df_refe_mon_cropgroups_landcover_2018['MON_LC_GROUP'] == 'MON_LC_GRASSES']['MON_CROPGROUP'].unique().tolist() # number=1
    ALL_MON_CROPGROUPS_IN_MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_LIST = df_refe_mon_cropgroups_landcover_2018[df_refe_mon_cropgroups_landcover_2018['MON_LC_GROUP'] == 'MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS']['MON_CROPGROUP'].unique().tolist() # number=1
    ALL_MON_CROPGROUPS_IN_MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_NS_LIST = df_refe_mon_cropgroups_landcover_2018[df_refe_mon_cropgroups_landcover_2018['MON_LC_GROUP'] == 'MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_NS']['MON_CROPGROUP'].unique().tolist() # number=1! not 4, missing  'MON_BOOMKWEEK','MON_CONTAINERS','MON_STAL_SER']#
    ALL_MON_CROPGROUPS_IN_MON_LC_INELIGIBLE_LIST = df_refe_mon_cropgroups_landcover_2018[df_refe_mon_cropgroups_landcover_2018['MON_LC_GROUP'] == 'MON_LC_INELIGIBLE']['MON_CROPGROUP'].unique().tolist() # number=1
    ALL_MON_CROPGROUPS_IN_MON_LC_UNKNOWN_LIST  = df_refe_mon_cropgroups_landcover_2018[df_refe_mon_cropgroups_landcover_2018['MON_LC_GROUP'] == 'MON_LC_UNKNOWN']['MON_CROPGROUP'].unique().tolist() # number =1
    # STEP 2_2
    
    ## STEP 1_4 Get list from column headers (40 +2 )of the predictions proba input
    PREDICTION_COLUMN_LIST= df_parcel_predictions_proba.columns.tolist() 

    # STEP 2_2 Reducing the FULL Lists (82) to list with the values (40) that occur in the prediction table
    MON_CROPGROUPS_IN_MON_LC_ARABLE_LIST=set(ALL_MON_CROPGROUPS_IN_MON_LC_ARABLE_LIST).intersection(PREDICTION_COLUMN_LIST) # number = 30
    MON_CROPGROUPS_IN_MON_LC_FABACEAE_LIST=set(ALL_MON_CROPGROUPS_IN_MON_LC_FABACEAE_LIST).intersection(PREDICTION_COLUMN_LIST) # number =4
    MON_CROPGROUPS_IN_MON_LC_FALLOW_LIST=set(ALL_MON_CROPGROUPS_IN_MON_LC_FALLOW_LIST).intersection(PREDICTION_COLUMN_LIST) # number = 1
    MON_CROPGROUPS_IN_MON_LC_GRASSES_LIST=set(ALL_MON_CROPGROUPS_IN_MON_LC_GRASSES_LIST).intersection(PREDICTION_COLUMN_LIST) # number = 1
    MON_CROPGROUPS_IN_MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_LIST=set(ALL_MON_CROPGROUPS_IN_MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_LIST).intersection(PREDICTION_COLUMN_LIST) # number = 1
    MON_CROPGROUPS_IN_MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_NS_LIST=set(ALL_MON_CROPGROUPS_IN_MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_NS_LIST).intersection(PREDICTION_COLUMN_LIST) # number = 1
    MON_CROPGROUPS_IN_MON_LC_INELIGIBLE_LIST=set(ALL_MON_CROPGROUPS_IN_MON_LC_INELIGIBLE_LIST).intersection(PREDICTION_COLUMN_LIST) # number = 1
    MON_CROPGROUPS_IN_MON_LC_UNKNOWN_LIST=set(ALL_MON_CROPGROUPS_IN_MON_LC_UNKNOWN_LIST).intersection(PREDICTION_COLUMN_LIST) # number = 1

    # STEP 2_3
    MON_LC_GROUP_SUM_LIST = ["MON_LC_ARABLE", "MON_LC_FABACEAE","MON_LC_FALLOW", "MON_LC_GRASSES", "MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS", "MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_NS", "MON_LC_INELIGIBLE", "MON_LC_UNKNOWN"] # checksum

    # STEP3 
    df_parcel_predictions_proba['MON_LC_ARABLE'] = df_parcel_predictions_proba[MON_CROPGROUPS_IN_MON_LC_ARABLE_LIST].sum(axis=1)
    df_parcel_predictions_proba['MON_LC_FABACEAE'] = df_parcel_predictions_proba[MON_CROPGROUPS_IN_MON_LC_FABACEAE_LIST].sum(axis=1)
    df_parcel_predictions_proba['MON_LC_FALLOW'] = df_parcel_predictions_proba[MON_CROPGROUPS_IN_MON_LC_FALLOW_LIST].sum(axis=1)
    df_parcel_predictions_proba['MON_LC_GRASSES'] = df_parcel_predictions_proba[MON_CROPGROUPS_IN_MON_LC_GRASSES_LIST].sum(axis=1)
    df_parcel_predictions_proba['MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS'] = df_parcel_predictions_proba[MON_CROPGROUPS_IN_MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_LIST].sum(axis=1)
    df_parcel_predictions_proba['MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_NS'] = df_parcel_predictions_proba[MON_CROPGROUPS_IN_MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_NS_LIST].sum(axis=1)
    df_parcel_predictions_proba['MON_LC_INELIGIBLE'] = df_parcel_predictions_proba[MON_CROPGROUPS_IN_MON_LC_INELIGIBLE_LIST].sum(axis=1)
    df_parcel_predictions_proba['MON_LC_UNKNOWN'] = df_parcel_predictions_proba[MON_CROPGROUPS_IN_MON_LC_UNKNOWN_LIST].sum(axis=1)

    # STEP 3_2 also reclassify the classes colums !!

    df_parcel_predictions_proba.loc[df_parcel_predictions_proba["classname"].isin(list(MON_CROPGROUPS_IN_MON_LC_ARABLE_LIST)), 'classname'] = 'MON_LC_ARABLE'
    df_parcel_predictions_proba.loc[df_parcel_predictions_proba["classname"].isin(list(MON_CROPGROUPS_IN_MON_LC_FABACEAE_LIST)), 'classname'] = 'MON_LC_FABACEAE'
    df_parcel_predictions_proba.loc[df_parcel_predictions_proba["classname"].isin(list(MON_CROPGROUPS_IN_MON_LC_FALLOW_LIST)), 'classname'] = 'MON_LC_FALLOW'
    df_parcel_predictions_proba.loc[df_parcel_predictions_proba["classname"].isin(list(MON_CROPGROUPS_IN_MON_LC_GRASSES_LIST)), 'classname'] = 'MON_LC_GRASSES'
    df_parcel_predictions_proba.loc[df_parcel_predictions_proba["classname"].isin(list(MON_CROPGROUPS_IN_MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_LIST)), 'classname'] = 'MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS'
    df_parcel_predictions_proba.loc[df_parcel_predictions_proba["classname"].isin(list(MON_CROPGROUPS_IN_MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_NS_LIST)), 'classname'] = 'MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_NS'
    df_parcel_predictions_proba.loc[df_parcel_predictions_proba["classname"].isin(list(MON_CROPGROUPS_IN_MON_LC_INELIGIBLE_LIST)), 'classname'] = 'MON_LC_INELIGIBLE'
    df_parcel_predictions_proba.loc[df_parcel_predictions_proba["classname"].isin(list(MON_CROPGROUPS_IN_MON_LC_UNKNOWN_LIST)), 'classname'] = 'MON_LC_UNKNOWN'

    ## STEP 4_1 
    df_output_predictions_fortop3=df_parcel_predictions_proba[['UID']+['classname'] + MON_LC_GROUP_SUM_LIST]
    

    #Write the output file
    pdh.to_file(df_output_predictions_fortop3, output_predictions_proba_filepath)
    
    
    #"""


def calc_top3_and_consolidation(input_parcel_filepath: str,
                                input_parcel_probabilities_filepath: str,
                                output_predictions_filepath: str,
                                force: bool = False):

    if(force is False
       and os.path.exists(output_predictions_filepath)):
        logger.warning(f"calc_top3_and_consolidation: output file exist and force is False, so stop: {output_predictions_filepath}")
        return

    # Read input files
    df_input_parcel = pdh.read_file(input_parcel_filepath)
    df_proba = pdh.read_file(input_parcel_probabilities_filepath)

    # Calculate the top 3 predictions
    df_top3 = _get_top_3_prediction(df_proba)

    # Add the consolidated prediction
    # TODO: try to rewrite using native pandas commands to improve performance
    def calculate_consolidated_prediction(row):
        # For some reason the row['pred2_prob'] is sometimes seen as string, and so 2* gives a
        # repetition of the string value instead of a mathematic multiplication... so cast to float!

        # float(row['pred1_prob']) > 30
        if ((float(row['pred1_prob']) >= 2.0 * float(row['pred2_prob']))):
            return row[conf.columns['prediction']]
        else:
            return 'DOUBT'

    values = df_top3.apply(calculate_consolidated_prediction, axis=1)
    df_top3.insert(loc=2, column=conf.columns['prediction_withdoubt'], value=values)

    # Make sure all input parcels are in the output. If there was no prediction, it means that there
    # was no data available for a classification, so set prediction to NODATA
    df_top3.set_index(conf.columns['id'], inplace=True)
    if df_input_parcel.index.name != conf.columns['id']:
        df_input_parcel.set_index(conf.columns['id'], inplace=True)

    # Add a column with the prediction status... and all parcels in df_top3 got a prediction
    df_top3[conf.columns['prediction_status']] = 'OK'
    df_top3.loc[(df_top3[conf.columns['prediction_withdoubt']] == 'DOUBT'),
                conf.columns['prediction_status']] = 'DOUBT'

    cols_to_join = df_top3.columns.difference(df_input_parcel.columns)
    df_pred = df_input_parcel.join(df_top3[cols_to_join], how='left')
    df_pred[conf.columns['prediction']].fillna('NODATA', inplace=True)
    df_pred[conf.columns['prediction_withdoubt']].fillna('NODATA', inplace=True)
    df_pred[conf.columns['prediction_status']].fillna('NODATA', inplace=True)

    logger.info(f"Columns of df_pred: {df_pred.columns}")

    # Now calculate the full consolidated prediction: 
    #    * Can be doubt if probability too low
    #    * Parcels with too few pixels don't have a good accuracy and give many alfa errors...
    df_pred[conf.columns['prediction_cons']] = df_pred[conf.columns['prediction_withdoubt']]
    df_pred.loc[(df_pred[conf.columns['pixcount_s1s2']] <= conf.marker.getint('min_nb_pixels'))
                 & (df_pred[conf.columns['prediction_status']] != 'NODATA')
                 & (df_pred[conf.columns['prediction_status']] != 'DOUBT'),
                [conf.columns['prediction_cons'], conf.columns['prediction_status']]] = 'NOT_ENOUGH_PIXELS'

    # Set the prediction status for classes that should be ignored
    df_pred.loc[df_pred[conf.columns['class']].isin(conf.marker.getlist('classes_to_ignore_for_train')), 
                [conf.columns['prediction_status']]] = 'UNKNOWN'
    df_pred.loc[df_pred[conf.columns['class']].isin(conf.marker.getlist('classes_to_ignore')), 
                [conf.columns['prediction_status']]] = df_pred[conf.columns['class']]

    # Calculate consequences for the predictions
    logger.info("Calculate the consequences for the predictions")

    def add_prediction_conclusion(in_df,
                                  new_columnname, 
                                  prediction_column_to_use,
                                  detailed: bool):
        """
        Calculate the "conclusions" for the predictions 

        REMARK: calculating it like this, using native pandas operations, is 300 times faster than
                using DataFrame.apply() with a function!!!
        """
        # Add the new column with a fixed value first 
        in_df[new_columnname] = 'UNDEFINED'

        # Get a list of the classes to ignore
        all_classes_to_ignore = conf.marker.getlist('classes_to_ignore_for_train') + conf.marker.getlist('classes_to_ignore')

        # Some conclusions are different is detailed info is asked...
        if detailed == True:
            # Parcels that were ignored for trainig and/or prediction, get an ignore conclusion
            in_df.loc[in_df[conf.columns['class']].isin(all_classes_to_ignore),
                      new_columnname] = 'IGNORE:INPUTCLASSNAME=' + in_df[conf.columns['class']].map(str)
            # If conclusion still UNDEFINED, check if doubt 
            in_df.loc[(in_df[new_columnname] == 'UNDEFINED')
                    & (in_df[prediction_column_to_use].isin(['DOUBT', 'NODATA', 'NOT_ENOUGH_PIXELS'])),
                    new_columnname] = 'DOUBT:REASON=' + in_df[prediction_column_to_use].map(str)
        else:
            # Parcels that were ignored for trainig and/or prediction, get an ignore conclusion
            in_df.loc[in_df[conf.columns['class']].isin(all_classes_to_ignore),
                      new_columnname] = 'IGNORE'
            # If conclusion still UNDEFINED, check if doubt 
            in_df.loc[(in_df[new_columnname] == 'UNDEFINED')
                    & (in_df[prediction_column_to_use].isin(['DOUBT', 'NODATA', 'NOT_ENOUGH_PIXELS'])),
                    new_columnname] = 'DOUBT'

        # If conclusion still UNDEFINED, check if prediction equals the input class 
        in_df.loc[(in_df[new_columnname] == 'UNDEFINED')
                & (in_df[conf.columns['class']] == in_df[prediction_column_to_use]),
                new_columnname] = 'OK:PREDICTION=INPUT_CLASS'
        # If conclusion still UNDEFINED, prediction is different from input 
        in_df.loc[in_df[new_columnname] == 'UNDEFINED',
                new_columnname] = 'NOK:PREDICTION<>INPUT_CLASS'

    # Calculate detailed conclusions
    add_prediction_conclusion(in_df=df_pred,
                              new_columnname=conf.columns['prediction_conclusion_detail'],
                              prediction_column_to_use=conf.columns['prediction'],
                              detailed=True)
    add_prediction_conclusion(in_df=df_pred,
                              new_columnname=conf.columns['prediction_conclusion_detail_withdoubt'],
                              prediction_column_to_use=conf.columns['prediction_withdoubt'],
                              detailed=True)
    add_prediction_conclusion(in_df=df_pred,
                              new_columnname=conf.columns['prediction_conclusion_detail_cons'],
                              prediction_column_to_use=conf.columns['prediction_cons'],
                              detailed=True)

    # Calculate general conclusions for cons as well
    add_prediction_conclusion(in_df=df_pred,
                              new_columnname=conf.columns['prediction_conclusion_cons'],
                              prediction_column_to_use=conf.columns['prediction_cons'],
                              detailed=False)

    logger.info("Write final prediction data to file")
    pdh.to_file(df_pred, output_predictions_filepath)

def _get_top_3_prediction(df_probabilities):
    """ Returns the top 3 predictions for each parcel.

    The return value will be a dataset with the following columns:
        - id_column: id of the parcel as in the input
        - class_column: class of the parcel as in the input
        - prediction_cons_column: the consolidated prediction, will be 'DOUBT'
          if the prediction had a relatively low probability.
        - prediction_columns: prediction with the highest probability
        - pred1_prob: probability of the best prediction
        - pred2
        - pred2_prob
        - pred3
        - pred3_prob
    """

    logger.info("get_top_3_predictions: start")
    df_probabilities_tmp = df_probabilities.copy()
    for column in df_probabilities_tmp.columns:
        if column in conf.preprocess.getlist('dedicated_data_columns'):
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
    id_class_top3 = np.concatenate([df_probabilities[[conf.columns['id'], conf.columns['class']]].values, top3_pred]
                                   , axis=1)

    # Convert to dataframe and return
    df_top3 = pd.DataFrame(id_class_top3,
                           columns=[conf.columns['id'], conf.columns['class'],
                                    conf.columns['prediction'], 'pred2', 'pred3',
                                    'pred1_prob', 'pred2_prob', 'pred3_prob'])

    return df_top3