# import PreProcessing as pp
import metricModules as mm
# import requests
import re
import logging
import json
import sys
import pandas as pd
import numpy as np

uniqueID = input('Please input a unique identifier for your dataset: ')


fileNamesList = ['../data/PuneAQM_Data_2022/puneAQM.json']

# fileNamesList = ['../data/split_file_0.json',
#               '../data/split_file_1.json',
#               '../data/split_file_2.json',
#               '../data/split_file_3.json',
#               '../data/split_file_4.json',
#               '../data/split_file_5.json',
#               '../data/split_file_6.json',
#               '../data/split_file_7.json',
#               '../data/split_file_8.json',
#               '../data/split_file_9.json']
# preprocessing
# reading the file

lengthlist = []
iatRegularityMetricScore = []  
iatOutliersMetricScore = []
outlierNum = []
# totalDataPackets = []
duplicatesMetricScore = []
formatMetricScore = []
addnlAttrMetricScore = []
compMetricScore = []
avgDataQualityScore = []
dfAggregate = pd.DataFrame()
dfAggIAT = pd.DataFrame()

for chunk in fileNamesList:
    #reading the chunk and schema
    lengthlist.append(chunk)
    with open(chunk, "r") as jfile:
            jsonDataDict = json.load(jfile)
    dataframe = pd.json_normalize(jsonDataDict)
    pd.set_option('mode.chained_assignment', None)

    schema = '../schemas/schema_EnvAQM.json'

    # aggregating dataframe to compute duplicates across entire dataset instead of across each chunk
    if len(lengthlist) == 1:
        dfGrouped = dataframe.groupby(uniqueID).agg(
             count = ('observationDateTime', 'count'),
             nunique = ('observationDateTime', 'nunique')).reset_index()
        dfAggregate = dfGrouped

    #aggregating dataframe for IAT Regularity metric computation
        dfIAT = mm.preProcess(dataframe, uniqueID)
        # dfIAT = dfIAT.groupby(uniqueID).agg({'IAT': 'first'})
        dfAggIAT = dfIAT
        print(dfAggIAT)
    elif len(lengthlist) > 1:
        dfGrouped = dataframe.groupby(uniqueID).agg(
             count = ('observationDateTime', 'count'),
             nunique = ('observationDateTime', 'nunique')).reset_index()
        dfGrouped = pd.concat([dfGrouped, dfAggregate], ignore_index = True)
        dfGroupedCombined = dfGrouped.groupby([uniqueID,'nunique']).agg({
                                                    'count':'sum'}).reset_index()
        dfAggregate = dfGroupedCombined

        #aggregating for IAT Regularity metric computation
        dfIATCombined = mm.preProcess(dataframe, uniqueID)
        # dfIATCombined = dfIATCombined.groupby(uniqueID).agg({'IAT': 'first'})
        dfIATCombined = pd.concat([dfIATCombined, dfAggIAT], ignore_index = True)
         
        dfAggIAT = dfIATCombined
        # print(dfAggIAT)
        
    #duplicate metric computation
    # dfAggregate[uniqueID] = dfAggregate[uniqueID].str[-4:]
    # //TODO: make function
    dupeNum = dfAggregate['count'].sum() - dfAggregate['nunique'].sum()
    totalPackets = dfAggregate['count'].sum()
    duplicatesMetricScore = (1 - (dupeNum/totalPackets))
    # print(dupeNum, totalPackets, duplicatesMetricScore)

    #iat outlier metric computations
    # iatOutliersMetricScore = mm.iatOutliersMetric(dfAggIAT)


    dataframeUID = mm.preProcess(dataframe, uniqueID)
    dfOutlier = dfAggIAT
    print(dfOutlier)
    # outlierNum.append(mm.iatOutliersMetric(dfOutlier))
    # outlierNumSum = sum(outlierNum)
    # print(outlierNumSum)
    # print(totalPackets)
    # iatOutliersMetricScore = np.round((1 - (outlierNumSum/totalPackets)), 3)
    iatOutliersMetricScore = mm.iatOutliersMetric(dfOutlier)
    # print(dataframeUID)

    # metric computation old
    #iat related metrics
    iatRegularityMetricScore.append(mm.iatRegularityMetric(dataframeUID))
    # iatOutliersMetricScore.append(mm.iatOutliersMetric(dataframeUID))

    # duplicatesMetricScore.append(mm.duplicatesMetric(dataframe))

    #schema related metrics
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    schemaFile = schema
    dataFile = chunk
    #Load the data file
    with open(dataFile, "r") as f:
        data = json.load(f)

    #Load the schema file
    with open(schemaFile, "r") as f1:
        schema = json.load(f1)


    #####################################################################
    # Format Validity for attributes for which data formats are provided
    # in the Schema
    # For this metric we will ignore
    #   1. errors that occur because "required" fields are not there
    #   2. errors that occur because additional fields are present
    #
    #####################################################################

    # Remove Required properties and Additional Properties from Schema

    schema['additionalProperties'] = False

    # del schema['additionalProperties']
    # del schema['required']

    num_samples, err_count, err_data_arr, add_err_count, req_err_cnt = mm.validate_data_with_schema(dataFile, schema)

    format_adherence_metric = (1 - (err_count-add_err_count-req_err_cnt)/num_samples)
    #logging.debug(err_data_arr)
    logging.debug('###########################################################################')
    logging.debug("Total Samples: " + str(num_samples))
    logging.debug("Total Format Errors: " + str(err_count))
    logging.debug("Format Adherence Metric: " + str(format_adherence_metric))
    logging.debug('###########################################################################')


    #######################################################################
    # Unknown data attribute metric: Fraction of data points for
    # which contain only known fields:(1 - fraction(datapoints that contain
    # unknown attribtues))
    # in the Schema
    # For this metric we will ignore
    #   1. errors that occur because of format issues
    #   2. errors that occur because of required fields not present
    #
    #######################################################################

    logging.debug(err_data_arr)
    unknown_fields_absent_metric = 1 - add_err_count/num_samples
    logging.debug("Total samples: " + str(num_samples))
    logging.debug("Total Additional Fields Error Count: " + str(add_err_count))
    logging.debug("Unknown_Attributes_Absent_Metric: " + str(unknown_fields_absent_metric))

    #######################################################################
    # One by one check the required properties are present in packets or
    # not  
    #######################################################################

    with open(schemaFile, "r") as f1:
        schema = json.load(f1)

    del schema['additionalProperties']
    req = schema['required']
    logging.debug(len(req))
    missing_attr = {}
    completeness_metric = 0
    num_samples, total_missing_count = mm.validate_requiredFields(dataFile, req)

    logging.debug("Total missing count: " + str(total_missing_count))


    completeness_metric = 1 - total_missing_count/(num_samples*len(req))

    logging.debug('###########################################################################')
    logging.debug('##### Total Missing Fields Count for Required fields #######')
    logging.debug("Total samples: " + str(num_samples))
    logging.debug("Attribute_Completeness_Metric: "+str(completeness_metric))
    logging.debug('###########################################################################')


# output formatting

    compMetricScore.append(round(completeness_metric, 3))
    formatMetricScore.append(round(format_adherence_metric, 3))
    addnlAttrMetricScore.append(round(unknown_fields_absent_metric, 3))
    avgDataQualityScore.append(round((iatRegularityMetricScore[len(lengthlist) - 1] + iatOutliersMetricScore + duplicatesMetricScore + compMetricScore[len(lengthlist) - 1] + formatMetricScore[len(lengthlist) - 1] + addnlAttrMetricScore[len(lengthlist) - 1])/6, 3))

    # logging.info('################## Final Metrics #########################################')
    # logging.info('#')
    # logging.info("Inter-Arrival Time Regularity Metric: " + str(iatRegularityMetricScore[len(lengthlist) - 1]))
    # logging.info('#')
    # logging.info("Inter-Arrival Time Outliers Metric: " + str(iatOutliersMetricScore[len(lengthlist) - 1]))
    # logging.info('#')
    # logging.info("Absence of Duplicate Values Metric: " + str(duplicatesMetricScore))
    # logging.info('#')
    # logging.info("Adherence to Attribute Format Metric: " + str(formatMetricScore[len(lengthlist) - 1]))
    # logging.info('#')
    # logging.info("Absence of Unknown Attributes Metric: " + str(addnlAttrMetricScore[len(lengthlist) - 1]))
    # logging.info('#')
    # logging.info("Adherence to Mandatory Attributes Metric: " + str(compMetricScore[len(lengthlist) - 1]))
    # logging.info('###########################################################################')
    # logging.info('#')
    # logging.info("Average Data Quality Score for this chunk: " + str(avgDataQualityScore[len(lengthlist) - 1]))
    # logging.info('#')
    # logging.info('###########################################################################')


iatRegularityMetricScore = np.mean(iatRegularityMetricScore)
iatOutliersMetricScore = np.mean(iatOutliersMetricScore)
duplicatesMetricScore = np.round(duplicatesMetricScore, 3)
formatMetricScore = np.mean(formatMetricScore)
addnlAttrMetricScore = np.mean(addnlAttrMetricScore)
compMetricScore = np.mean(compMetricScore)
avgDataQualityScore = np.mean(avgDataQualityScore)

logging.info('################## Final Metrics #########################################')
logging.info('#')
logging.info("Inter-Arrival Time Regularity Metric: " + str(iatRegularityMetricScore))
logging.info('#')
logging.info("Inter-Arrival Time Outliers Metric: " + str(iatOutliersMetricScore))
logging.info('#')
logging.info("Absence of Duplicate Values Metric: " + str(duplicatesMetricScore))
logging.info('#')
logging.info("Adherence to Attribute Format Metric: " + str(formatMetricScore))
logging.info('#')
logging.info("Absence of Unknown Attributes Metric: " + str(addnlAttrMetricScore))
logging.info('#')
logging.info("Adherence to Mandatory Attributes Metric: " + str(compMetricScore))
logging.info('###########################################################################')
logging.info('#')
logging.info("Average Data Quality Score for the dataset: " + str(avgDataQualityScore))
logging.info('#')
logging.info('###########################################################################')
