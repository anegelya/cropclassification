# -*- coding: utf-8 -*-
"""
Created on Mon May  6 22:01:43 2019

@author: janant
"""

import os
from pandas import DataFrame, read_csv
import pandas as pd 
import numpy as np
import global_settings as gs
import classification_preprocess as class_pre


# STEP 1 READ FILES
#  VARIABLES
#****************************************************************
# the input file "MONGROEPEN_new3 .csv"  is the result of a query on the PRI DB, 
# it is a classification of all (292 or 293) unique cropnames / classnames of the campagne year 2018 
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
#
# UPDATE "X:\Monitoring\Markers\playground\_refe\refe_mon_cropgroups_landcover_2018.csv"
# 288 klassen 
#
# YEAR                  288 non-null int64
# CROPCODE              288 non-null int64
# CROP_DESC             288 non-null object
# DIV_CROPGROUP         214 non-null object
# DIV_CROPGROUP_DESC    214 non-null object
# MON_CROPGROUP         288 non-null object
# MON_CROPGROUP_DESC    288 non-null object
# MON_LC_GROUP          288 non-null object
#****************************************************************
## STEP 1_1A 
# read the MONGROEPEN_new3csv into a dataframe
df_mongroep= pd.read_csv(r"X:\Monitoring\Markers\PlayGround\JanAnt\InputData\MONGROEPEN_new3.csv",sep=';')
##  STEP 1_1B read the refe_mon_cropgroups_landcover_2018.csv into a dataframe
df_refe_mon_cropgroups_landcover_2018=pd.read_csv(r"X:\Monitoring\Markers\playground\_refe\refe_mon_cropgroups_landcover_2018.csv",sep=',', encoding='cp1252')
## STEP 1_2A make a dataframe with an erase of the duplicates in the MON_groep values, 
# to have a table of 89 unique MON_groep in mon lC groups for later use, thus reducing the 299 records to 89
df_monlcgroep=df_mongroep.drop_duplicates(subset ='MON_groep')
##  STEP 1_2B make a dataframe with an erase of the duplicates in the MON_CROPGROUP values, 
# to have a table of 82 unique mongroups in MON_LC_GROUP for later use
df_refe_mon_cropgroups_landcovergroups_2018=df_refe_mon_cropgroups_landcover_2018.drop_duplicates(subset ='MON_CROPGROUP')

##To make resume table for internal use
#refetest2=refetest.groupby('MON_LC_GROUP')['MON_CROPGROUP'].agg(', '.join)
#refetest2=df_refe_mon_cropgroups_landcovergroups_2018.groupby('MON_LC_GROUP')['MON_CROPGROUP'].agg(','.join)
# refetest2=refetest.groupby('MON_CROPGROUP')['CROP_DESC'].agg(', '.join)
# refetest2=refetest.groupby('MON_CROPGROUP')['CROPCODE'].apply(str).agg(', '.join)
# refetest2=refetest.groupby('MON_CROPGROUP').agg({'CROPCODE':'sum'})
# refetest2=refetest.groupby('MON_CROPGROUP').agg({'CROP_DESC':'sum','CROP_DESC':', '.join }) *
# refetest2=refetest.groupby('MON_LC_GROUP')['MON_CROPGROUP'].agg(', '.join) *
# refetest2=refetest.groupby('MON_LC_GROUP')['MON_CROPGROUP'].unique().agg(', '.join) **
# refetest2.to_csv(r"X:\Monitoring\Markers\PlayGround\JanAnt\temp\MONGROEPENlijst02.csv",sep=';') **
# refetest4=refetest.groupby('MON_CROPGROUP')['CROP_DESC'].unique().agg(', '.join) **
# refetest4.to_csv(r"X:\Monitoring\Markers\PlayGround\JanAnt\temp\MONGROEPENlijst04.csv",sep=';') **
# list(refetest['CROPCODE'].astype(str).unique())
# refetest5=refetest.assign(CROPCODEstring=lambda refetest:refetest.CROPCODE)
# refetest4=refetest5.groupby('MON_CROPGROUP')['CROPCODEstring'].unique().agg(', '.join) **
# refetest_cropcodes_in_MON_CROPGROUP=refetest5.groupby('MON_CROPGROUP')['CROPCODEstring'].unique().agg(', '.join)
# refetest_MON_CROPGROUP_in_MON_LC_GROUP=refetest5.groupby('MON_LC_GROUP')['MON_CROPGROUP'].unique().agg(', '.join)
# refetest_MON_CROPGROUP_in_MON_LC_GROUP.to_csv(r"X:\Monitoring\Markers\PlayGround\JanAnt\temp\refetest_MON_CROPGROUP_in_MON_LC_GROUP.csv",sep=';')
# refetest_cropcodes_in_MON_CROPGROUP.to_csv(r"X:\Monitoring\Markers\PlayGround\JanAnt\temp\refetest_cropcodes_in_MON_CROPGROUP.csv",sep=';')
# refetest_cropcodesdesc_in_MON_CROPGROUP=refetest5.groupby('MON_CROPGROUP')['CROP_DESC'].unique().agg(', '.join)
# refetest_cropcodesdesc_in_MON_CROPGROUP.to_csv(r"X:\Monitoring\Markers\PlayGround\JanAnt\temp\refetest_cropcodesdesc_in_MON_CROPGROUP.csv",sep=';')
# merge01=pd.merge(refetest,refetest_MON_CROPGROUP_in_MON_LC_GROUP, how='outer', on='MON_LC_GROUP' )
# merge02=pd.merge(merge01,refetest_cropcodes_in_MON_CROPGROUP, how='outer', left_on='MON_CROPGROUP_x', right_on='MON_CROPGROUP')
# merge02.to_csv(r"X:\Monitoring\Markers\PlayGround\JanAnt\temp\refetest_merge.csv",sep=';')
# merge03=pd.merge(merge02,refetest_cropcodesdesc_in_MON_CROPGROUP, how='outer', left_on='CROP_DESC', right_on='CROP_DESC')
#
#
#
# STEP 1_3 Read out the temp predictions output csv file of the classtype_to_prepare = 'MONITORING_CROPGROUPS' process , that file is > 400MB 
# This file commes out of classification_sklearn.py 
df_output_predictions= pd.read_csv(r"X:\Monitoring\Markers\PlayGround\JanAnt\temp\temp_output_predictions.csv",sep=',' )

# STEP 1_4 Get list from column headers, 41 in total 3 info colums + there are 38 prediction classes /some of them are technical like MON_CONTAINERS, MON_BONEN_WIKKEN , MON_BOOMKWEEK, MON_STAL_SER..
PREDICTION_COLUMN_LIST= df_output_predictions.columns.tolist() 


# STEP 2 A
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

# STEP 2 B
#****************************************************************
# A. Preprocessing to aggregate the colums for summation
# 1. Making lists of all (82+7) possible values based on the mongroep input tabel(= FULL_***_LIST)
# 2  Reducing the list of values by intersecting with occuring vallues in the prediction table (***_LIST)
#  Since some classes like MON_AUBERGINES, MON_AZALEA have not enough elements they are directly set to 'UNKNOWN', 
#  so they don't occur in the prediction csv
# the sum of the 8 different lists is 86, since the prediction list cannot contain the 3 special** codes
#****************************************************************
ALL_MON_CROPGROUPS_IN_MON_LC_ARABLE_LIST= df_refe_mon_cropgroups_landcover_2018[df_refe_mon_cropgroups_landcover_2018['MON_LC_GROUP'] == 'MON_LC_ARABLE']['MON_CROPGROUP'].unique().tolist() # number=67
ALL_MON_CROPGROUPS_IN_MON_LC_FABACEAE_LIST = df_refe_mon_cropgroups_landcover_2018[df_refe_mon_cropgroups_landcover_2018['MON_LC_GROUP'] == 'MON_LC_FABACEAE']['MON_CROPGROUP'].unique().tolist() # number=8! not 9  NOT MON_BONEN_WIKKEN !!!
ALL_MON_CROPGROUPS_IN_MON_LC_FALLOW_LIST = df_refe_mon_cropgroups_landcover_2018[df_refe_mon_cropgroups_landcover_2018['MON_LC_GROUP'] == 'MON_LC_FALLOW']['MON_CROPGROUP'].unique().tolist() # number=2
ALL_MON_CROPGROUPS_IN_MON_LC_GRASSES_LIST = df_refe_mon_cropgroups_landcover_2018[df_refe_mon_cropgroups_landcover_2018['MON_LC_GROUP'] == 'MON_LC_GRASSES']['MON_CROPGROUP'].unique().tolist() # number=1
ALL_MON_CROPGROUPS_IN_MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_LIST = df_refe_mon_cropgroups_landcover_2018[df_refe_mon_cropgroups_landcover_2018['MON_LC_GROUP'] == 'MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS']['MON_CROPGROUP'].unique().tolist() # number=1
ALL_MON_CROPGROUPS_IN_MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_NS_LIST = df_refe_mon_cropgroups_landcover_2018[df_refe_mon_cropgroups_landcover_2018['MON_LC_GROUP'] == 'MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_NS']['MON_CROPGROUP'].unique().tolist() # number=1! not 4, missing  'MON_BOOMKWEEK','MON_CONTAINERS','MON_STAL_SER']
ALL_MON_CROPGROUPS_IN_MON_LC_INELIGIBLE_LIST = df_refe_mon_cropgroups_landcover_2018[df_refe_mon_cropgroups_landcover_2018['MON_LC_GROUP'] == 'MON_LC_INELIGIBLE']['MON_CROPGROUP'].unique().tolist() # number=1
ALL_MON_CROPGROUPS_IN_MON_LC_UNKNOWN_LIST  = df_refe_mon_cropgroups_landcover_2018[df_refe_mon_cropgroups_landcover_2018['MON_LC_GROUP'] == 'MON_LC_UNKNOWN']['MON_CROPGROUP'].unique().tolist() # number =1
# Reducing the FULL Lists to list with values that occur in the prediction table , sum is 38
MON_CROPGROUPS_IN_MON_LC_ARABLE_LIST=  set(ALL_MON_CROPGROUPS_IN_MON_LC_ARABLE_LIST).intersection(PREDICTION_COLUMN_LIST) # number = 26
MON_CROPGROUPS_IN_MON_LC_FABACEAE_LIST =  set(ALL_MON_CROPGROUPS_IN_MON_LC_FABACEAE_LIST).intersection(PREDICTION_COLUMN_LIST) # number =3
MON_CROPGROUPS_IN_MON_LC_FALLOW_LIST=  set(ALL_MON_CROPGROUPS_IN_MON_LC_FALLOW_LIST).intersection(PREDICTION_COLUMN_LIST) # number = 1
MON_CROPGROUPS_IN_MON_LC_GRASSES_LIST = set(ALL_MON_CROPGROUPS_IN_MON_LC_GRASSES_LIST).intersection(PREDICTION_COLUMN_LIST) # number = 1
MON_CROPGROUPS_IN_MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_LIST = set(ALL_MON_CROPGROUPS_IN_MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_LIST).intersection(PREDICTION_COLUMN_LIST) # number =
MON_CROPGROUPS_IN_MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_NS_LIST = set(ALL_MON_CROPGROUPS_IN_MON_LC_IGNORE_DIFFICULT_PERMANENT_CLASS_NS_LIST).intersection(PREDICTION_COLUMN_LIST) # number =
MON_CROPGROUPS_IN_MON_LC_INELIGIBLE_LIST = set(ALL_MON_CROPGROUPS_IN_MON_LC_INELIGIBLE_LIST).intersection(PREDICTION_COLUMN_LIST) # number = 1
MON_CROPGROUPS_IN_MON_LC_UNKNOWN_LIST =set(ALL_MON_CROPGROUPS_IN_MON_LC_UNKNOWN_LIST).intersection(PREDICTION_COLUMN_LIST) # number =
# Extra SUM List of the SUMS of the MON_LC_GROUPS SUM for check reasons
SUM_SUM_LIST = ["ARABLE_SUM", "FABACEAE_SUM","FALLOW_SUM", "GRASSES_SUM", "IGNORE_DIFFICULT_PERMANENT_CLASS_SUM", "IGNORE_DIFFICULT_PERMANENT_CLASS_NS_SUM", "INELIGIBLE_SUM", "UNKNOWN_SUM"] # checksum

#****************************************************************
# STEP3 A
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
