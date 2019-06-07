# -*- coding: utf-8 -*-
"""
Module that implements the classification logic.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.svm import SVC

import cropclassification.helpers.config_helper as conf
import cropclassification.helpers.pandas_helper as pdh

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
import skl2onnx.common.data_types as onnxcommon
import onnxruntime as onxx_rt

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def train(df_train: pd.DataFrame,
          output_classifier_filepath: str):
    """
    Train a classifier and output the trained classifier to the output file.

    Args
        df_train: pandas DataFrame containing the train data. Columns:
            * global_settings.id_column: the id of the parcel
            * global_settings.class_column: the class of the parcel
            * ... all columns that will be used as classification data
        output_classifier_filepath: the filepath where the classifier can be written
    """

    # Split the input dataframe in one with the train classes and one with the train data
    df_train_classes = df_train[conf.columns['class']]
    cols_to_keep = df_train.columns.difference([conf.columns['id'], conf.columns['class']])
    df_train_data = df_train[cols_to_keep]

    logger.info(f"Train file processed and rows with missing data removed, data shape: {df_train_data.shape}, labels shape: {df_train_classes.shape}")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info(f"Resulting Columns for training data: {df_train_data.columns}")

    # Using almost all defaults for the classifier seems to work best...
    logger.info('Start training')
    classifier_type_lower = conf.classifier['classifier_type'].lower()
    if classifier_type_lower == 'randomforest':
        n_estimators = conf.classifier.getint('randomforest_n_estimators')
        max_depth = conf.classifier.getint('randomforest_max_depth')
        classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    elif classifier_type_lower == 'nearestneighbor':
        n_neighbors = conf.classifier.getint('nearestneighbor_n_neighbors')
        weights = conf.classifier['nearestneighbor_weights']
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, n_jobs=-1)
    elif classifier_type_lower == 'multilayer_perceptron':
        hidden_layer_sizes = tuple(conf.classifier.getlistint('multilayer_perceptron_hidden_layer_sizes'))
        max_iter = conf.classifier.getint('multilayer_perceptron_max_iter')
        classifier = MLPClassifier(max_iter=max_iter, hidden_layer_sizes=hidden_layer_sizes)
    elif classifier_type_lower == 'svm':
        # cache_size=1000 (MB) should speed up training
        # probability=True is necessary to be able to use predict_proba
        classifier = SVC(C=1.0, gamma='auto', 
                probability=True, cache_size=1000)
    else:
        message = f"Unsupported classifier in conf.classifier['classifier_type']: {conf.classifier['classifier_type']}"
        logger.critical(message)
        raise Exception(message)

    logger.info(f"Start fitting classifier:\n{classifier}")
    classifier.fit(df_train_data, df_train_classes)

    # write the learned model to a file -- onnx
    datatype = onnxcommon.guess_data_type(df_train_data)
    onnx = convert_sklearn(classifier, 'cropclassification', [('input', FloatTensorType([1, len(datatype)]))])
    with open(output_classifier_filepath.replace('.pkl', '.onnx'), "wb") as f:
        f.write(onnx.SerializeToString())

    # Write the learned model to a file... -- pkl
    logger.info(f"Write the learned model file to {output_classifier_filepath}")
    joblib.dump(classifier, output_classifier_filepath)    

def predict_proba(df_input_parcel: pd.DataFrame,
                  input_classifier_filepath: str,
                  output_parcel_predictions_filepath: str) -> pd.DataFrame:
    """
    Predict the probabilities for all input data using the classifier provided and write it
    to the output file.

    Args
        df_input_parcel: pandas DataFrame containing the data to classify. Columns:
            * global_settings.id_column: the id of the parcel.
            * global_settings.class_column: the class of the parcel. Isn't really used.
            * ... all columns that will be used as classification data.
        output_classifier_filepath: the filepath where the classifier can be written.
    """

    # Some basic checks that input is ok
    df_input_parcel.reset_index(inplace=True)
    if(conf.columns['id'] not in df_input_parcel.columns
       or conf.columns['class'] not in df_input_parcel.columns):
        message = f"Columns {conf.columns['id']} and {conf.columns['class']} are mandatory for input parameter df_input!"
        logger.critical(message)
        raise Exception(message)

    # Now do final preparation for the classification
    df_input_classes = df_input_parcel[conf.columns['class']]
    cols_to_keep = df_input_parcel.columns.difference([conf.columns['id'], conf.columns['class']])
    df_input_data = df_input_parcel[cols_to_keep]

    logger.info(f"Train file processed and rows with missing data removed, data shape: {df_input_data.shape}, labels shape: {df_input_classes.shape}")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info(f"Resulting Columns for training data: {df_input_data.columns}")

    #cols = [conf.columns['id'], conf.columns['class']]

    # Load the classifier
    if (conf.classifier["use_onnx"]):
        sess = onxx_rt.InferenceSession(input_classifier_filepath.replace(".pkl", ".onnx"))    
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[1].name # 0: predict, 1: predict_proba

        class_proba = {}
        input_data_reshaped_array = np.reshape(df_input_data.values, (len(df_input_data.index), len(df_input_data.columns)))
        for i, x in enumerate(input_data_reshaped_array[:10]):
            pred_onx = sess.run([label_name], {input_name: x})[0]
            class_proba[i] = pred_onx[0]

        #cols.extend(class_proba[0][0].keys())
        class_proba_df = pd.DataFrame(class_proba)
        
    else:
        classifier = joblib.load(input_classifier_filepath)
        logger.info(f"Classifier has the following columns: {classifier.classes_}")

        logger.info(f"Predict classes with probabilities: {len(df_input_parcel)} rows")
        class_proba = classifier.predict_proba(df_input_data)
        logger.info(f"Predict classes with probabilities ready")

        #cols.extend(classifier.classes_)
        class_proba_df = pd.DataFrame(class_proba, columns=classifier.classes_)

    # Convert probabilities to dataframe, combine with input data and write to file
    #id_class_proba = np.concatenate([df_input_parcel[[conf.columns['id'], conf.columns['class']]].values[:10], class_proba], axis=1)
    #df_proba = class_proba_df.join(df_input_parcel[[conf.columns['id'], conf.columns['class']]], how='inner')
    df_proba = class_proba_df
    df_proba.insert(0, df_input_parcel[conf.columns['class']])
    df_proba.insert(0, df_input_parcel[conf.columns['id']])
    #df_proba = pd.DataFrame(id_class_proba, columns=cols)

    # If output path provided, write results
    if output_parcel_predictions_filepath:
        pdh.to_file(df_proba, output_parcel_predictions_filepath)

    return df_proba

# If the script is run directly...
if __name__ == "__main__":
    logger.critical('Not implemented exception!')
    raise Exception('Not implemented')
