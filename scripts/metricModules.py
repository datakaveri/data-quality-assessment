import scipy.stats as sps
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ijson
import jsonschema
import fastjsonschema
# import json
# import sys
import logging
import pandas as pd
import re
# import os

# data handling functions
#interarrival time creation
def preProcess(df, uniqueID):
    # df['observationDateTime'] =  pd.to_datetime(df['observationDateTime'], origin = 'unix', unit = 'ms')
    df['observationDateTime'] =  pd.to_datetime(df['observationDateTime'])
    df = df[['observationDateTime', uniqueID]]
    df.sort_values(by = [uniqueID, 'observationDateTime'], inplace = True, ascending = [True, True])
    df['IAT'] = df['observationDateTime'].diff().dt.total_seconds()
    df['IAT'] = df['IAT'][(df['IAT']>0)]
    # df.sort_values(by = 'IAT', inplace = True, ascending = True)
    df = df.drop(['observationDateTime'], axis = 1)
    df = df.reset_index(drop = True) 
    df.dropna()
    return(df)

# metric computation functions
##############################
# interarrival time regularity metric
def computeModeDeviation(dataframe):
    modeValue = sps.mode(dataframe)[0][0]
    deviations = abs(dataframe - modeValue)
    modeDeviation = deviations.sum() / len(dataframe)
    return modeDeviation

def iatRegularityMetricOld(dataframe):
    # dataframe['id'] = dataframe['id'].str[-4:]

    grouped = dataframe.groupby('id')
    # grouped = dataframe
    # print(grouped)
    result = grouped['IAT'].apply(computeModeDeviation).reset_index()
    result.columns = ['id', 'mode_deviation']

    # Normalize the mode deviation values between 0 and 1
    scaler = MinMaxScaler()
    result['normalized_mode_deviation'] = scaler.fit_transform(result[['mode_deviation']])
    iatRegularityMetricScore = 1 - np.mean(result['normalized_mode_deviation'])
    # print(iatRegularityMetricScore)
    return round(iatRegularityMetricScore, 3)

def iatRegularityMetric(dataframe):
    modeValue = dataframe['IAT'].mode()[0]
    goodCount = 0
    badCount = 0
    count = 0
    for iat in dataframe['IAT']:
        RAE_i = (np.abs(iat - modeValue))/modeValue 
        if RAE_i <= 0.5:
            goodCount += 1 - 2*RAE_i
            count += 1
        else:
            # print(badCount, iat, RAE_i)
            badCount += 2*RAE_i
        totalCount = count + badCount
    iatRegularityMetricScore = goodCount/totalCount
    # print(goodCount, badCount, count, totalCount, iatRegularityMetricScore)
    return round(iatRegularityMetricScore, 3)

# interarrival time outliers metric
def iatOutliersMetric(dataframe):
    df = dataframe
    data = df['IAT'].dropna()
    # print(Q1, Q3)
    # adaptive threshold using regular z-score
    mode = sps.mode(data)[0]
    mad = np.median(np.abs(data - mode))
    print(mode)
    modified_z_scores = (0.6745 * (data - mode)) / mad # applicable to normal symmetric distribution
    threshold_mod_z_score = 3.5 # as recommended by Iglewicz and Hoaglin
    #defining fences
    outliers = [x for x in df['IAT'] if ((0.6745 * (x - mode)) / mad) > threshold_mod_z_score] 
    outlierNumber = len(outliers)
    totalDataPackets = len(df)
    # print(outlierNumber, totalDataPackets)
    iatMetricOutlierScore = 1 - (outlierNumber/totalDataPackets)
    # print(iatMetricOutlierScore)
    # return outlierNumber      
    return round(iatMetricOutlierScore, 3)

# duplicate detection metric
# must be called before inter-arrival time creation
def duplicatesMetric(df, input1, input2):
    dupeCount = len(df) - len(df.drop_duplicates(subset = [input1, input2]))
    totalDataPackets = len(df)
    duplicatesMetricScore = 1 - dupeCount/totalDataPackets
    return round(duplicatesMetricScore,3)

# schema validation metrics

#Expecting 'data' to be a list
def validate_data_with_schema(dataF, schema):
    num_samples = 0
    err_count = 0
    additional_prop_err_count = 0
    req_prop_err_count = 0
    err_data_arr = []

    with open(dataF, "r") as f:
        for record in ijson.items(f, "item"):
            num_samples = num_samples+1
            data_packet = record
            try:
                fastjsonschema.validate(schema, data_packet)

            except fastjsonschema.exceptions.JsonSchemaValueException as errV:
                logging.debug ("Validation Error Occured")
               #v = jsonschema.Draft7Validator(schema, types=(), format_checker=None)
                v = jsonschema.Draft7Validator(schema)
                errors = list(v.iter_errors(data_packet))
                # errors = sorted(v.iter_errors(data_packet), key=lambda e: e.path)
                if len(errors) > 0:
                    err_count = err_count + 1
                
               #err_data_arr.append(data_packet)
               #To track if 'Additional Properties' error occured
                flag_0 = 0
               #To track if 'Required Properties' error occured
                flag_1 = 0
                for error in errors:
                    logging.debug (error.message)
                    z = re.match("(Additional properties)", error.message)
                    if z:
                      #logging.debug(z.groups())
                        flag_0 = 1

                    z =  error.message.split(' ')
                    if z[-1] == 'property' and z[-2] == 'required' :
                        flag_1 = flag_1+1

                additional_prop_err_count = additional_prop_err_count + flag_0
                req_prop_err_count = req_prop_err_count + flag_1
            except jsonschema.exceptions.SchemaError as errS:
                logging.debug ("Schema Error Occured")
                logging.debug (errS.message)
        
    return num_samples, err_count, err_data_arr, additional_prop_err_count, req_prop_err_count

def validate_requiredFields(dataF, setReqd):
    num_samples = 0
    num_missing_prop = 0
    with open(dataF, "r") as f:
       #Read each record instead of reading all at a time
       for record in ijson.items(f, "item"):
            num_samples = num_samples+1
           #setRecd = record.keys()
            setRecd = []
           #Null value detection. Null values are considered as not received attribute
            for attr in record.keys():
                if record[attr] is None:
                    logging.debug("Received a Null Value for attribute: " + attr)
                else:
                    setRecd.append(attr)
            diffSet = set(setReqd) - set(setRecd)
            logging.debug("Difference from Required Fields for this packet: "+str(diffSet))
            num_missing_prop = num_missing_prop + len(diffSet)
    return num_samples, num_missing_prop
