import metrics.PreProcessing as pp
import metrics.data_preprocessing as dp
import metrics.iat_regularity_metrics as irm
import metrics.iat_outliers_metrics as iom
import metrics.duplicates_metrics as dm
import metrics.schema_validation_metrics as svm
import metrics.required_fields_validation as rfv
import metrics.pdf_generation as pdf_gen
import json
import sys
import re
import logging
import os

def main():
    """
    Main function to execute data quality assessment pipeline.
    
    Orchestrates the complete data quality analysis workflow including:
    - Configuration file reading and data loading
    - Schema validation 
    - Data preprocessing and outlier removal
    - Calculation of various data quality metrics
    - Generation of visualizations and reports
    
    The function interactively prompts user for configuration file name,
    schema validation preferences, and PDF report generation choices.
    
    Returns:
        None: Outputs JSON report file and optional PDF report to disk
    """
    configFile = "../config/" + input("Enter the name of the configuration file: ")
    configDict, dfRaw, input1, input2, datasetName, fileName, URL, alpha, schema = (
        pp.readFile(configFile)
    )
    print(fileName)
    print(datasetName)
    startTime, endTime, startMonth, endMonth, startYear, endYear = pp.timeRange(dfRaw)
    numPackets = dfRaw.shape[0]

    # Validating Data against Schema
    schemaProvision = input("Do you have a schema to validate the data against? [y/n]: ")
    schemaInputValidity = 0
    while schemaInputValidity == 0:
        if schemaProvision == "y":
            logging.basicConfig(stream=sys.stderr, level=logging.INFO)

            schemaFile = "../schemas/" + configDict["schemaFileName"]
            print(schemaFile)
            dataFile = fileName

            # Load the data file
            with open(dataFile, "r") as f:
                data = json.load(f)

            # Load the schema file
            with open(schemaFile, "r") as f1:
                schema = json.load(f1)

            # Format Validity for attributes for which data formats are provided in the Schema
            # For this metric we will ignore:
            # 1. errors that occur because "required" fields are not there
            # 2. errors that occur because additional fields are present
            # Remove Required properties and Additional Properties from Schema
            schema["additionalProperties"] = False

            num_samples, err_count, err_data_arr, add_err_count, req_err_cnt = (
                svm.validate_data_with_schema(dataFile, schema)
            )

            format_adherence_metric = (
                1 - (err_count - add_err_count - req_err_cnt) / num_samples
            )
            logging.debug(
                "###########################################################################"
            )
            logging.debug("Total Samples: " + str(num_samples))
            logging.debug("Total Format Errors: " + str(err_count))
            logging.debug("Format Adherence Metric: " + str(format_adherence_metric))
            logging.debug(
                "###########################################################################"
            )

            # Unknown data attribute metric: Fraction of data points for which contain only known fields
            # (1 - fraction(datapoints that contain unknown attribtues))
            # For this metric we will ignore:
            # 1. errors that occur because of format issues
            # 2. errors that occur because of required fields not present
            logging.debug(err_data_arr)
            unknown_fields_absent_metric = 1 - add_err_count / num_samples
            logging.debug("Total samples: " + str(num_samples))
            logging.debug("Total Additional Fields Error Count: " + str(add_err_count))
            logging.debug(
                "Unknown_Attributes_Absent_Metric: " + str(unknown_fields_absent_metric)
            )

            # Check the required properties are present in packets or not
            with open(schemaFile, "r") as f1:
                schema = json.load(f1)

            del schema["additionalProperties"]
            req = schema["required"]
            logging.debug(len(req))
            missing_attr = {}
            completeness_metric = 0
            num_samples, total_missing_count = rfv.validate_requiredFields(dataFile, req)

            logging.debug("Total missing count: " + str(total_missing_count))

            completeness_metric = 1 - total_missing_count / (num_samples * len(req))

            logging.debug(
                "###########################################################################"
            )
            logging.debug("##### Total Missing Fields Count for Required fields #######")
            logging.debug("Total samples: " + str(num_samples))
            logging.debug("Attribute_Completeness_Metric: " + str(completeness_metric))
            logging.debug(
                "###########################################################################"
            )
            schemaInputValidity = 1
        elif schemaProvision == "n":
            format_adherence_metric = 0
            unknown_fields_absent_metric = 0
            completeness_metric = 0
            schemaInputValidity = 1
        else:
            print("Please provide a valid input [y/n]: ")
            schemaProvision = input(
                "Do you have a schema to validate the data against? [y/n]: "
            )

    # Running data preprocessing functions
    # Dropping duplicates
    dfDropped, dupeCount = pp.dropDupes(dfRaw, input1, input2)
    # Cleaning dataframe
    dfClean = dp.preProcess(dfDropped, input1, input2)

    # Removing outliers
    dfInliers, lowerOutliers, upperOutliers = pp.outRemove(dfClean, datasetName, input1)

    # Data statistics before/after removing outliers
    (
        meanStatOut,
        medianStatOut,
        modeStatOut,
        stdStatOut,
        varianceStatOut,
        skewStatOut,
        kurtosisStatOut,
    ) = pp.dataStats(dfClean)
    (
        meanStatIn,
        medianStatIn,
        modeStatIn,
        stdStatIn,
        varianceStatIn,
        skewStatIn,
        kurtosisStatIn,
    ) = pp.dataStats(dfInliers)

    # Running functions that are used to calculate the metric scores
    regularityMetricScore = irm.iatRegularityMetric(dfClean)
    outliersMetricScore = iom.iatOutliersMetric(dfClean)
    dupeMetricScore = dm.duplicatesMetric(dfRaw, input1, input2)
    compMetricScore = round(completeness_metric, 3)
    formatMetricScore = round(format_adherence_metric, 3)
    addnlAttrMetricScore = round(unknown_fields_absent_metric, 3)
    avgDataQualityScore = round(
        (
            regularityMetricScore
            + outliersMetricScore
            + dupeMetricScore
            + compMetricScore
            + formatMetricScore
            + addnlAttrMetricScore
        )
        / 6,
        3,
    )
    avgDataQualityPercent = round(avgDataQualityScore * 100, 3)

    logging.info(
        "################## Final Metrics #########################################"
    )
    logging.info("#")
    logging.info("Inter-Arrival Time Regularity: " + str(regularityMetricScore))
    logging.info("#")
    logging.info("Inter-Arrival Time Outliers Metric: " + str(outliersMetricScore))
    logging.info("#")
    logging.info("Absence of Duplicate Values Metric: " + str(dupeMetricScore))
    logging.info("#")
    logging.info("Adherence to Attribute Format Metric: " + str(formatMetricScore))
    logging.info("#")
    logging.info("Absence of Unknown Attributes Metric: " + str(addnlAttrMetricScore))
    logging.info("#")
    logging.info("Adherence to Mandatory Attributes Metric: " + str(compMetricScore))
    logging.info(
        "###########################################################################"
    )
    logging.info("#")
    logging.info("Average Data Quality Score: " + str(avgDataQualityScore))
    logging.info("#")
    logging.info(
        "###########################################################################"
    )

    # Naming dataframes for plot file naming
    dfRaw.name = "raw"
    dfDropped.name = "dropped"
    dfClean.name = "clean"
    dfInliers.name = "inliers"

    # Generating visualizations for the PDF in order of appearance in the report
    # DQ overview horizontal bars
    pp.bars(regularityMetricScore, "regularity")
    pp.bars(outliersMetricScore, "outliers")
    pp.bars(dupeMetricScore, "dupe")
    pp.bars(compMetricScore, "comp")
    pp.bars(formatMetricScore, "format")
    pp.bars(addnlAttrMetricScore, "addnl")

    # Half pie charts
    pp.gaugePlot(regularityMetricScore, "regularityMetricScore")
    pp.gaugePlot(outliersMetricScore, "outliersMetricScore")
    pp.gaugePlot(dupeMetricScore, "dupeMetricScore")
    pp.gaugePlot(compMetricScore, "compMetricScore")
    pp.gaugePlot(formatMetricScore, "formatMetricScore")
    pp.gaugePlot(addnlAttrMetricScore, "addnlAttrMetricScore")

    # Radar chart
    pp.radarChart(
        regularityMetricScore,
        outliersMetricScore,
        dupeMetricScore,
        compMetricScore,
        formatMetricScore,
        addnlAttrMetricScore,
    )

    # Inter-arrival time boxplots and histogram
    pp.IAThist(dfClean)
    pp.boxPlot(dfClean, fileName, input1)

    # Without outliers boxplots and histograms
    pp.IAThist(dfInliers)
    pp.boxPlot(dfInliers, fileName, input1)
    pp.normalFitPlot(dfClean)

    # Outliers boxplot
    pp.outliersPlot(dfClean)

    # Duplicates bar chart
    pp.plotDupesID(dfRaw, dfDropped, input1)
    pp.piePlot(dfRaw, dfClean, "dupe")

    # Skewness and kurtosis calculated for inlier values only
    muFitIn, stdFitIn = pp.normalFitPlot(dfInliers)

    # Skewness and kurtosis calculated for outlier values included
    muFitOut, stdFitOut = pp.normalFitPlot(dfClean)

    # Get file name without extension for naming output files
    fileNameNoExt = os.path.splitext(os.path.basename(fileName))[0]

    # Ask user if they want to generate PDF report
    pdf_choice = input("\nDo you want to generate a PDF report? [y/n]: ").lower().strip()
    
    if pdf_choice == 'y':
        print("\nGenerating PDF report...")
        
        # Generate PDF report
        pdf_filename = f"{fileNameNoExt}_DQReport.pdf"
        pdf_gen.create_analytics_report_schema(
            filename=pdf_filename,
            datasetName=datasetName,
            URL=URL,
            numPackets=numPackets,
            startTime=startTime,
            endTime=endTime,
            regularityMetricScore=regularityMetricScore,
            outliersMetricScore=outliersMetricScore,
            dupeMetricScore=dupeMetricScore,
            compMetricScore=compMetricScore,
            formatMetricScore=formatMetricScore,
            addnlAttrMetricScore=addnlAttrMetricScore,
            avgDataQualityScore=avgDataQualityScore,
            avgDataQualityPercent=avgDataQualityPercent,
            input1=input1,
            input2=input2,
            dupeCount=dupeCount
        )
        
        print(f"PDF report generated successfully: {pdf_filename}")
    else:
        print("Skipping PDF report generation.")

    # Output Report as JSON
    outputParamFV = {
        "fileName": datasetName,
        "startTime": str(startTime),
        "endTime": str(endTime),
        "No. of data packets": numPackets,
        "avgDataQualityScore": avgDataQualityScore,
        "IAT Regularity": {
            "overallValue": regularityMetricScore,
            "type": "number",
            "metricLabel": "IAT Regularity Metric",
            "metricMessage": f"For this dataset, the inter-arrival time regularity metric value is {regularityMetricScore}",
            "description": "This metric is rated on a scale between 0 & 1; computes the output of the equation (1 - ((No.of data packets outside the bounds)/(Total no. of data packets)). These bounds are defined by the value of alpha and the formula (mode +/- (alpha*mode)). The overall metric score is formed from an average of the three scores obtained from three values of alpha.",
        },
        "IATOutliers": {
            "value": outliersMetricScore,
            "type": "number",
            "metricLabel": "IAT Outlier Metric",
            "metricMessage": f"For this dataset, the inter-arrival time outliers metric score is {outliersMetricScore}.",
            "description": "This metric is rated on a scale between 0 & 1; it is computed using the modified Z-score method and is calculated as (1-(No. of outliers/No. of data packets))",
        },
        "Absence of Duplicate Values": {
            "value": dupeMetricScore,
            "deduplicationAttributes": [input1, input2],
            "type": "number",
            "metricLabel": "Duplicate Value Metric",
            "metricMessage": f"For this dataset, the duplicate value metric score is: {dupeMetricScore}.",
            "description": "This metric is rated on a scale between 0 & 1; it is computed using the formula (1 - (No. of duplicate data packets/total no. of data packets).",
        },
        "Adherence to Attribute Format": {
            "value": format_adherence_metric,
            "type": "number",
            "metricLabel": "Format Adherence Metric",
            "metricMessage": "For this dataset, "
            + str(format_adherence_metric)
            + " is the format adherence",
            "description": "The metric is rated on a scale between 0 & 1; computed using the formula (1 - (no. of format validity errors/total no. of data packets)).",
        },
        "Absence of Unknown Attributes": {
            "value": unknown_fields_absent_metric,
            "type": "number",
            "metricLabel": "Unknown Attributes Metric",
            "metricMessage": "For this dataset, "
            + str(unknown_fields_absent_metric)
            + " is the value of the  additional fields absent metric.",
            "description": "The metric is rated on a scale between 0 & 1; computed as (1 - r) where r is the ratio of packets with unknown attributes to the total number of packets.",
        },
        "Adherence to Mandatory Attributes": {
            "value": completeness_metric,
            "type": "number",
            "metricLabel": "Completeness Metric",
            "metricMessage": "For this dataset, "
            + str(completeness_metric)
            + " is the value of the adherence to mandatory attributes metric.",
            "description": "The metric is rated on a scale between 0 & 1; It is computed as follows: For each mandatory attribute, i, compute r(i) as the ratio of packets in which attribute i is missing. Then output 1 - average(r(i)) where the average is taken over all mandatory attributes.",
        },
    }
    
    myJSON = json.dumps(outputParamFV, indent=4)
    filename = fileNameNoExt + "_Report.json"
    jsonpath = os.path.join("../outputReports/", filename)

    with open(jsonpath, "w+") as jsonfile:
        jsonfile.write(myJSON)
        print(f"JSON output file successfully created: {filename}")

if __name__ == "__main__":
    main()