## -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:00:37 2019

@author: 
"""

from pandas import DataFrame, read_csv
#import matplotlib.pyplot as plt
import pandas as pd 
MON_LC_ALL37_LIST = ["MON_AARDAPPELEN", "MON_ASPERGE", "MON_BIETEN", "MON_BONEN_WIKKEN", "MON_BOOMKWEEK", "MON_CHRYSANTEN", "MON_CICHOREIACHTIGEN", "MON_CONTAINERS", "MON_COURGETTES_EN_POMPOENEN", "MON_ERWTEN", "MON_GERST", "MON_GERST_WINTER", "MON_GRASSEN", "MON_HAVER", "MON_HAVER_WINTER", "MON_KERST", "MON_KOOLACHTIGEN", "MON_KOOLZAAD_WINTER", "MON_LOOKACHTIGEN", "MON_LUZERNE", "MON_MAIS", "MON_NATUUR", "MON_PASTINAAK", "MON_PETERSELIE", "MON_RABARBER", "MON_ROGGE_WINTER", "MON_SCHORSENEREN", "MON_SELDERACHTIGEN", "MON_SPELT", "MON_SPINAZIE", "MON_STAL_SER", "MON_TAGETES_AFRIKAANTJE", "MON_TARWE", "MON_TARWE_WINTER", "MON_TRITICALE", "MON_VLAS", "MON_WORTEL"]
MON_LC_FABACEAE_LIST_theoretisch=["MON_ANDERE_SUBSID_GEWASSEN",	"MON_BONEN", "MON_BONEN_WIKKEN", "MON_ERWTEN", "MON_KLAVER", "MON_LUPINEN", "MON_LUZERNE", "MON_MENGSEL", "MON_SOJABONEN", "MON_WIKKEN"]
MON_LC_FALLOW_LIST_theoretisch=["MON_BRAAK"]
MON_LC_ARABLE_LIST_theoretisch=["MON_(KNOL)VENKEL", "MON_AARDAPPELEN", "MON_AARDBEIEN", "MON_ANDERE_SUBSID_GEWASSEN", 
                    "MON_ARTISJOKKEN", "MON_ASPERGE", "MON_AUBERGINES", "MON_AZALEA", "MON_BASILICUM", 
                    "MON_BEGONIA", "MON_BIETEN", "MON_BLOEM", "MON_BOEKWEIT", "MON_CHRYSANTEN", 
                    "MON_CICHOREIACHTIGEN", "MON_COURGETTES_EN_POMPOENEN", "MON_ENGELWORTEL", 
                    "MON_FACELIA", "MON_GELE_MOSTERD", "MON_GERST", "MON_GERST_WINTER", "MON_HAVER", 
                    "MON_HAVER_WINTER", "MON_HENNEP", "MON_KERVEL", "MON_KOMKOMMERKRUID", "MON_KOMKOMMERS", 
                    "MON_KOOLACHTIGEN", "MON_KOOLRAAPACHTIGEN", "MON_KOOLZAAD", "MON_KOOLZAAD_WINTER", "MON_LOOKACHTIGEN", 
                    "MON_MAIS", "MON_MARIADISTEL", "MON_MENGSEL", "MON_MOSTERD", "MON_MUSKAATPOMPOENEN", "MON_NYGER", 
                    "MON_PANICUM", "MON_PAPRIKAS", "MON_PASTINAAK", "MON_PETERSELIE", "MON_PHALARIS", "MON_QUINOA", 
                    "MON_RAAPACHTIGEN", "MON_RAAPACHTIGEN_WINTER", "MON_RABARBER", "MON_RADIJSACHTIGEN", "MON_ROGGE", 
                    "MON_ROGGE_WINTER", "MON_RUCOLA", "MON_SCHORSENEREN", "MON_SELDERACHTIGEN", "MON_SLA", 
                    "MON_SOEDANGRAS", "MON_SORGHUM", "MON_SPELT", "MON_SPINAZIE", "MON_TABAK", 
                    "MON_TAGETES_AFRIKAANTJE", "MON_TARWE", "MON_TARWE_WINTER", "MON_TOMATEN", 
                    "MON_TRITICALE", "MON_VELDSLA", "MON_VLAS", "MON_VLINDERBLOEMIGEN", 
                    "MON_VOEDERRAPEN", "MON_WORTEL", 
                    "MON_ZONNEBLOEM", "MON_ZWAARDHERIK" ]



# LISTS VARIABLES
IGNORE_DIFFICULT_PERMANENT_CLASS_LIST =["MON_BOOMKWEEK", "MON_CONTAINERS", "MON_STAL_SER"]
MON_LC_ARABLE_LIST= ["MON_AARDAPPELEN", "MON_ASPERGE", "MON_BIETEN", "MON_CHRYSANTEN", "MON_CICHOREIACHTIGEN", "MON_COURGETTES_EN_POMPOENEN", "MON_GERST", "MON_GERST_WINTER", "MON_HAVER", "MON_HAVER_WINTER", "MON_KOOLACHTIGEN", "MON_KOOLZAAD_WINTER", "MON_LOOKACHTIGEN", "MON_MAIS", "MON_PASTINAAK", "MON_PETERSELIE", "MON_RABARBER", "MON_ROGGE_WINTER", "MON_SCHORSENEREN", "MON_SELDERACHTIGEN", "MON_SPELT", "MON_SPINAZIE", "MON_TAGETES_AFRIKAANTJE", "MON_TARWE", "MON_TARWE_WINTER", "MON_TRITICALE", "MON_VLAS", "MON_WORTEL"]
MON_LC_FABACEAE_LIST = ["MON_BONEN_WIKKEN",  "MON_ERWTEN",  "MON_LUZERNE"]
MON_LC_GRASSEN_LIST =  ["MON_GRASSEN"]
MON_LC_INELIGIBLE_LIST = [ "MON_KERST", "MON_NATUUR"]
SUM_SUM_LIST = ["IGNORE_SUM", "ARABLE_SUM","FABACEAE_SUM", "GRASSEN_SUM", "INELIGIBLE_SUM"]
#MON_LC_INELIGIBLE=["MON_BRAAK"] #(81	Braakliggend land zonder minimale activiteit)
#UNKNOWN = ["MON_GRASSEN"] #  9	Onverharde landingsbaan of veiligheidszones op vliegvelden

#data = pd.read_csv(r"X:\Monitoring\Markers\PlayGround\JanAnt\2018_class_maincrops_mon\Run_003\BEFL2018_bufm10_weekly_predict_all.csv",
df_classes = pd.read_csv(r"X:\Monitoring\Markers\PlayGround\JanAnt\InputData\LANDCOVERGROEPEN.csv", 
                         sep=';', 
                         encoding='ANSI')
'''
df_predict_all = pd.read_csv(r"X:\Monitoring\Markers\PlayGround\JanAnt\2018_class_maincrops_mon\Run_003\BEFL2018_bufm10_weekly_predict_all.csv", 
                         sep=',', 
                         encoding='ANSI')
'''
#df_temp= pd.read_csv(r"X:\Monitoring\Markers\PlayGround\JanAnt\temp\temp_output_predictions.csv")
df_temp_mini= pd.read_csv(r"X:\Monitoring\Markers\PlayGround\JanAnt\temp\temp_output_predictions_mini02.csv")


df_temp_mini['ARABLE_SUM'] = df_temp_mini[MON_LC_ARABLE_LIST].sum(axis=1)
df_temp_mini['FABACEAE_SUM'] = df_temp_mini[MON_LC_FABACEAE_LIST].sum(axis=1)
df_temp_mini['GRASSEN_SUM'] = df_temp_mini[MON_LC_GRASSEN_LIST].sum(axis=1)
df_temp_mini['INELIGIBLE_SUM'] = df_temp_mini[MON_LC_INELIGIBLE_LIST].sum(axis=1)
df_temp_mini['IGNORE_SUM'] = df_temp_mini[IGNORE_DIFFICULT_PERMANENT_CLASS_LIST].sum(axis=1)
df_temp_mini['SUM_SUM'] = df_temp_mini[SUM_SUM_LIST].sum(axis=1)
print (df_temp_mini.head(5))



#df_MONGROEPEN_CSV= pd.read_csv(r"X:\Monitoring\Markers\PlayGround\JanAnt\InputData\LANDCOVERGROEPEN.csv", names=['Gewas', 'Campagne', 'MON_groep','MON_LC_groep'])
#print (df_MONGROEPEN_CSV)
#print (df_MONGROEPEN_CSV.sort_values(by=['MON_groep']))
#print (df_MONGROEPEN_CSV.groupby(by=['MON_groep']))

#mon_lc_arable_list=["MON_(KNOL)VENKEL", "MON_AARDAPPELEN", "MON_AARDBEIEN", "MON_ANDERE_SUBSID_GEWASSEN", "MON_ARTISJOKKEN", "MON_ASPERGE", "MON_AUBERGINES", "MON_AZALEA", "MON_BASILICUM", "MON_BEGONIA", "MON_BIETEN", "MON_BLOEM", "MON_BOEKWEIT", "MON_CHRYSANTEN", "MON_CICHOREIACHTIGEN", "MON_COURGETTES_EN_POMPOENEN", "MON_ENGELWORTEL", "MON_FACELIA", "MON_GELE_MOSTERD", "MON_GERST", "MON_GERST_WINTER", "MON_HAVER", "MON_HAVER_WINTER", "MON_HENNEP", "MON_KERVEL", "MON_KOMKOMMERKRUID", "MON_KOMKOMMERS", "MON_KOOLACHTIGEN", "MON_KOOLRAAPACHTIGEN", "MON_KOOLZAAD", "MON_KOOLZAAD_WINTER", "MON_LOOKACHTIGEN", "MON_MAIS", "MON_MARIADISTEL", "MON_MENGSEL", "MON_MOSTERD", "MON_MUSKAATPOMPOENEN", "MON_NYGER", "MON_PANICUM", "MON_PAPRIKAS", "MON_PASTINAAK", "MON_PETERSELIE", "MON_PHALARIS", "MON_QUINOA", "MON_RAAPACHTIGEN", "MON_RAAPACHTIGEN_WINTER", "MON_RABARBER", "MON_RADIJSACHTIGEN", "MON_ROGGE", "MON_ROGGE_WINTER", "MON_RUCOLA", "MON_SCHORSENEREN", "MON_SELDERACHTIGEN", "MON_SLA", "MON_SOEDANGRAS", "MON_SORGHUM", "MON_SPELT", "MON_SPINAZIE", "MON_TABAK", "MON_TAGETES_AFRIKAANTJE", "MON_TARWE", "MON_TARWE_WINTER", "MON_TOMATEN", "MON_TRITICALE", "MON_VELDSLA", "MON_VLAS", "MON_VLINDERBLOEMIGEN", "MON_VOEDERRAPEN", "MON_WORTEL", "MON_ZONNEBLOEM", "MON_ZWAARDHERIK]


#print('Max', data['pixcount'].max())
#print('Min', data['pixcount'].min())
#print('MONGROEP=', df_classes['MON_groep'])
#print('MAX=', df_classes['Gewas'].max())
#print('MIN=', df_predict_all['classname'] ,"-",  df_predict_all['classname'])
'''
df_join = df_predict_all.merge(df_classes,
                               how='left',
                               left_on=gs.class_column,
                               right_index=True,
                               right_on='MON_groep',
                               validate='many_to_one')
'''