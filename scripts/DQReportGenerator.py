import PreProcessing as pp
import metricModules as mm
import json
import sys
import re
import logging
import os
        
configFile = '../config/' + input('Enter the name of the configuration file: ')

configDict, dfRaw, input1, input2, datasetName, fileName, URL, schema, datasetType = pp.readFile(configFile)


print(fileName)
print(datasetName)
# print(os.path.splitext(os.path.basename(fileName))[0])

dateTimeColumn = 'observationDateTime'
if dateTimeColumn in dfRaw.columns:
    startTime, endTime, startMonth, endMonth, startYear, endYear = pp.timeRange(dfRaw)
else:
    pp.timeRange(dfRaw)
numPackets = dfRaw.shape[0]


# ### Validating Data against Schema
schemaProvision = input("Do you have a schema to validate the data against? [y/n]")
schemaInputValidity = 0
while schemaInputValidity == 0:
    if schemaProvision == 'y':
        #logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)

        schemaFile = '../schemas/' + configDict['schemaFileName']
        print(schemaFile)
        dataFile = fileName
        #Load the data file
        with open(dataFile, "r") as f:
            data = json.load(f)

        # Load the schema file
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
        # unknown attributes))
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
        schemaInputValidity = 1
    elif schemaProvision == 'n':
        format_adherence_metric = 0
        unknown_fields_absent_metric = 0
        completeness_metric = 0
        schemaInputValidity = 1
    else:
        print("Please provide a valid input [y/n]: ")
        schemaProvision = input("Do you have a schema to validate the data against? [y/n]")


# ### Running data preprocessing functions
#dropping duplicates
dfDropped, dupeCount = pp.dropDupes(dfRaw, input1, input2)
#cleaning dataframe
dateTimeColumn = 'observationDateTime'
if dateTimeColumn in dfRaw.columns:
    dfClean = pp.preProcess(dfDropped, input1, input2)



    #removing outliers
    dfInliers, lowerOutliers, upperOutliers = pp.outRemove(dfClean, datasetName, input1)
    # print(dfInliers, lowrOutliers, upperOutliers)

    #datastats before/after removing outliers
    meanStatOut, medianStatOut, modeStatOut, stdStatOut, varianceStatOut, skewStatOut, kurtosisStatOut = pp.dataStats(dfClean)
    meanStatIn, medianStatIn, modeStatIn, stdStatIn, varianceStatIn, skewStatIn, kurtosisStatIn = pp.dataStats(dfInliers)
    dfInliers.name = 'inliers'
else:
    dfClean = dfRaw
# print(dfClean['IAT'])
# dfClean.to_csv('dfCleantest.csv')


# dfInliers, lowerOutliers, upperOutliers = pp.outRemove(dfClean, datasetName, input1)
# print(lowerOutliers, upperOutliers)
# dfInliers.to_csv('dfInlierstest.csv')
# print(datasetName)


#running functions that are used to calcalate the metric scores
# regularityMetricScore, regularityValues, lowerRegularity, upperRegularity = pp.iatMetricRegularity(dfClean, alpha)
# regularityMetricScore = mm.iatRegularityMetric(dfClean)
# outliersMetricScore = mm.iatOutliersMetric(dfClean)
dupeMetricScore = mm.duplicatesMetric(dfRaw, input1, input2)
compMetricScore = round(completeness_metric, 3)
formatMetricScore = round(format_adherence_metric, 3)
addnlAttrMetricScore = round(unknown_fields_absent_metric, 3)
avgDataQualityScore = round((dupeMetricScore + compMetricScore + formatMetricScore + addnlAttrMetricScore)/4, 3)
avgDataQualityPercent = round(avgDataQualityScore*100,3)

logging.info('################## Final Metrics #########################################')
# logging.info('#')
# logging.info("Inter-Arrival Time Regularity: " + str(regularityMetricScore))
# logging.info('#')
# logging.info("Inter-Arrival Time Outliers Metric: " + str(outliersMetricScore))
logging.info('#')
logging.info("Absence of Duplicate Values Metric: " + str(dupeMetricScore))
logging.info('#')
logging.info("Adherence to Attribute Format Metric: " + str(formatMetricScore))
logging.info('#')
logging.info("Absence of Unknown Attributes Metric: " + str(addnlAttrMetricScore))
logging.info('#')
logging.info("Adherence to Mandatory Attributes Metric: " + str(compMetricScore))
logging.info('###########################################################################')
logging.info('#')
logging.info("Average Data Quality Score: " + str(avgDataQualityScore))
logging.info('#')
logging.info('###########################################################################')


#naming dataframes for plot file naming
dfRaw.name = 'raw'
dfDropped.name = 'dropped'
dfClean.name = 'clean'


# ### Generating visualizations for the PDF in order of appearance in the report
#DQ overview horizontal bars
# pp.bars(regularityMetricScore, 'regularity')
# pp.bars(outliersMetricScore, 'outliers')
pp.bars(dupeMetricScore, 'dupe')
pp.bars(compMetricScore, 'comp')
pp.bars(formatMetricScore, 'format')
pp.bars(addnlAttrMetricScore, 'addnl')

#half pie charts
# pp.gaugePlot(regularityMetricScore, 'regularityMetricScore')
# pp.gaugePlot(outliersMetricScore, 'outliersMetricScore')
pp.gaugePlot(dupeMetricScore, 'dupeMetricScore')
pp.gaugePlot(compMetricScore, 'compMetricScore')
pp.gaugePlot(formatMetricScore, 'formatMetricScore')
pp.gaugePlot(addnlAttrMetricScore, 'addnlAttrMetricScore')

#radar chart
pp.radarChart(
            # regularityMetricScore, 
            # outliersMetricScore, 
            dupeMetricScore, 
            compMetricScore,  
            formatMetricScore, 
            addnlAttrMetricScore
            )


# #interarrival time boxplots and histogram
# pp.IAThist(dfClean)
# pp.boxPlot(dfClean, fileName, input1)

#without outliers boxplots and histograms
# pp.IAThist(dfInliers)
# pp.boxPlot(dfInliers, fileName, input1)
# pp.normalFitPlot(dfClean)

#outliers boxplot
# pp.outliersPlot(dfClean)

#duplicates bar chart
pp.plotDupesID(dfRaw, dfDropped, input1)
# pp.plotDupes(dfRaw)
pp.piePlot(dfRaw, dfClean, 'dupe')



#data statistics plots
#correlation
# if 'AQM' in fileName:
#     pp.corr_heatmap(dfDropped)

#skewness and kurtosis calculated for inlier values only
# muFitIn, stdFitIn = pp.normalFitPlot(dfInliers)

#skewness and kurtosis calculated for outlier values included
# muFitOut, stdFitOut = pp.normalFitPlot(dfClean)

from fpdf import FPDF
from fpdf import *

# fileName = dataDict['dataFileNameJSON']
WIDTH = 210
HEIGHT = 297

# TEST_DATE = "10/20/20"
fileNameNoExt = os.path.splitext(os.path.basename(fileName))[0]

def create_title_card(pdf):
    # pdf.SetX(180)
    pdf.image("../plots/pretty/iudx.png", 10, 5, 35)
    pdf.set_font('times', 'b', 22)  
    pdf.set_x(60)
    pdf.write(5, "Data Quality Report")
    pdf.ln(10)
    # pdf.cell(20, 10, 'Title', 1, 1, 'R')
    pdf.set_font('times', '', 12)
    # pdf.write(4, f'Dataset: {dataSetName}')
    # pdf.set_x(120)
    pdf.ln(7)
    pdf.write(5, 'Dataset: ')
    pdf.set_text_color(r = 0, g = 0, b = 105)
    pdf.cell(10,5, f'{datasetName}', link = URL, align ='L')
    pdf.ln(7)
    pdf.set_text_color(r = 0, g = 0, b = 0)
    pdf.write(5, f'Data Type: {datasetType}')
    pdf.ln(7)
    # pdf.set_x(120)
    dateTimeColumn = 'observationDateTime'
    if dateTimeColumn in dfRaw.columns:
        pdf.cell(15, 5, f'Number of Data Packets: {numPackets}   |   Start Time: {startTime}   |   End Time: {endTime}')
    # pdf.ln(5)
    # pdf.set_x(120)
    # pdf.write(5, f'End Time: {endTime}')
    # pdf.ln(5)
    # pdf.set_x(120)
    # pdf.write(5, f'Number of Data Packets: {numPackets}')
    pdf.line(11, 46, 200, 46)


def create_heading(title, pdf):
    pdf.set_font('times', 'b', 18)
    pdf.ln(10)
    pdf.write(5, title)
    pdf.ln(5)
    pdf.set_font('times', '', 12)

        
class pdf(FPDF):
    def add_page(this,  same= True, orientation=''):
        FPDF.add_page(self, same= same, orientation=orientation)

    def footer(self):
            # Page number with condition isCover
            self.set_y(-15) # Position at 1.5 cm from bottom
            self.set_font('times', 'I', 8)
            self.cell(0, 10, 'Page  ' + str(self.page_no) + '  |  {nb}', 0, 0, 'C') 
        
        
# def create_analytics_report(filename=f"{fileName}_{startMonth}{startYear}{endMonth}{endYear}_DQReport.pdf"):
def create_analytics_report_schema(filename=f"{fileNameNoExt}_DQReport.pdf"):
    
    pdf = FPDF() # A4 (210 by 297 mm)

    ''' First Page '''
    pdf.add_page()
    #adding the banner/letterhead  
    
    create_title_card(pdf)
    create_heading('Overview', pdf)
    pdf.ln(5)
    
    #creating table of metric scores overview
    # todo add outliers/regularity differentiation
    data = [
            ['Metric','Score','Bar'],
            # ['Inter-Arrival Time Regularity',f'{regularityMetricScore}', ''],
            # ['Inter-Arrival Time Outliers', f'{outliersMetricScore}', ''],
            ['Duplicate Presence',f'{dupeMetricScore}', ''],
            ['Adherence to Attribute Format',f'{formatMetricScore}', ''],
            ['Absence of Unknown Attributes',f'{addnlAttrMetricScore}', ''],
            ['Adherence to Mandatory Attributes', f'{compMetricScore}','']
        ]
    
    # Text height is the same as current font size
    # Effective page width, or just epw
    epw = pdf.w - 2*pdf.l_margin

    # Set column width to 1/3 of effective page width to distribute content 
    # evenly across table and page
    col_width = epw/3
    th = pdf.font_size
    
    #logic included to bold only titles
    for row in data:
        for index, datum in enumerate(row):
            if datum == 'Metric' or datum == 'Score' or datum =='Bar':
                pdf.set_font('times', 'b', 15)
                if index == 1:
                    pdf.cell(col_width/2.7, 4*th, str(datum), border = 1, align = 'C')
                elif index == 2:
                    pdf.cell(col_width*1.5, 4*th, str(datum), border=1, align = 'C') 
                else:
                    pdf.cell(col_width+10, 4*th, str(datum), border=1, align = 'C')
            elif datum != 'Metric' or datum != 'Score' or datum != 'Bar':
                pdf.set_font('times', '', 12)
                if index == 1:
                    pdf.cell(col_width/2.7, 4*th, str(datum), border = 1, align = 'C')
                elif index == 2:
                    pdf.cell(col_width*1.5, 4*th, str(datum), border=1, align = 'C') 
                else:
                    pdf.cell(col_width+10, 4*th, str(datum), border=1, align = 'C')
        pdf.ln(4*th)

    #adding bars to the table
    # pdf.image("../plots/bars/regularitybar.png", 107, 76, 95)
    # pdf.image("../plots/bars/outliersbar.png", 107, 94, 95)
    pdf.image("../plots/bars/dupebar.png", 107, 76, 95)
    pdf.image("../plots/bars/formatbar.png", 107, 94, 95)
    pdf.image("../plots/bars/addnlbar.png", 107, 110, 95)
    pdf.image("../plots/bars/compbar.png", 107, 127, 95)
    
    pdf.ln(10)
    pdf.write(5, 'The Overall Data Quality Score of the dataset, computed by calculating an average of the above scores is:') 
    pdf.ln(10)
    pdf.set_font('times', 'b', 12)			
    pdf.write(5, f'{avgDataQualityScore}/1.00 or {avgDataQualityPercent}%')
    pdf.set_font('times', '', 12)
    pdf.ln(35)  
    
    #radar chart
    pdf.image("../plots/radarPlot.png", 110, 195, 95)
    
    
    pdf.write(5, 'This data quality assessment report shows the score for')
    pdf.ln(5)
    pdf.write(5, 'six metrics that contribute to data quality.')
    pdf.ln(10)
    pdf.write(5, 'The chart on the right shows an overview of the data') 
    pdf.ln(5)              
    pdf.write(5, 'quality of the dataset.')
    pdf.ln(10)
    pdf.write(5, 'In the following pages you can find a detailed description')
    pdf.ln(5)
    pdf.write(5, 'and breakdown of each of these metrics.')

    
    ''' Second Page '''
    # pdf.add_page()
    # pdf.image("plots/pretty/IUDXlogo.png", 0, 0, w = 60)
    pdf.ln(5)
    # create_heading('Inter-Arrival Time Regularity', pdf)
    # pdf.ln(5)
    # pdf.write(5, 'Inter-arrival time is defined as the time elapsed after the receipt of a data packet and until the receipt of the next packet. For sensor data, this is an important factor to evaluate as sensors are often configured to send data at specific time intervals.')
    # pdf.ln(5)
    # # create_heading('IAT Regularity', pdf)
    # pdf.image("../plots/donuts/regularityMetricScorePiePlot.png", x = 150, y = -5, w = 60)
    # pdf.ln(10)    
    # pdf.write(5, 'In order to compute this metric we analyse the deviation of each inter-arrival time from the mode. The assumption here is if most of the sensors are operating nominally most of the time, then the mode of the inter-arrival times will represent the expected nominal behaviour of the sensors. To compute this deviation, we define:')
    # pdf.ln(5)
    # pdf.image('../plots/equations/RAE_regularityMetric.png', x = 65, y = 80, w = 80)
    # pdf.ln(20)
    # pdf.write(5, 'Here, xi is the inter-arrival time, and x is the mode of the inter-arrival time. We consider an RAE value of 0.5 to be the crossover point between good and poor values of inter-arrival time, i.e. RAE > 0.5 is poor. We also want to penalise the score proportionately to the RAE value, meaning the greater the RAE value, the greater the penalty. RAE is thus bound as RAE belongs to [0, inf).')
    # pdf.ln(5)
    # pdf.write(5, 'The metric computation can also be represented as an equation:')
    # pdf.ln(5)
    # pdf.image('../plots/equations/regularityMetric.png', x = 60, y = 130, w = 80)
    # pdf.ln(40)
    # pdf.write(5, 'This represents the "badness" of the inter-arrival time when compared to the modal value. The further the inter-arrival time is from the mode, the greater the penalty contribution to the regularity score for that inter-arrival time. A value of 0.5 for RAE is chosen as the crossover point between "goodness" and "badness" of inter-arrival time as it represents a window of values corresponding to:')
    # # pdf.ln(10)
    # pdf.image('../plots/equations/mode_regularityMetric.png', x = 90, y = 190, w = 20)
    # pdf.ln(20)
    # pdf.write(5, 'A higher IAT Regularity score indicates lower dispersion of IAT values around the mode, and vice versa. A higher score indicates that there is a higher clustering of IAT values close to the mode of the sensor. This regularity is particularly important for time-critical applications where a consistent and predictable arrival pattern is desired. By evaluating the IAT Regularity metric, researchers can gain insights into the reliability and efficiency of the data transmission process in IoT networks, contributing to the optimization of various IoT applications and services.')
    # pdf.ln(10)
    
    #creating a table for the inter arrival time statistics
    # dataStats = [
    #             ['','Before Outlier Removal', 'After Removal of Outliers'],
    #             ['Mean', f'{meanStatOut}', f'{meanStatIn}'],
    #             ['Median', f'{medianStatOut}', f'{medianStatIn}'],
    #             ['Mode', f'{modeStatOut}', f'{modeStatIn}'],
    #             ['Standard Deviation', f'{stdStatOut}', f'{stdStatIn}'],
    #             ['Variance', f'{varianceStatOut}', f'{varianceStatIn}'],
    #             ['Skewness', f'{skewStatOut}', f'{skewStatIn}']
    #             ]
    
    # col_width = epw/3
    # #logic included to bold only titles
    # for row in dataStats:
    #     for index, datum in enumerate(row):
    #         if datum == 'Before Outlier Removal' or datum == 'After Removal of Outliers' or datum == 'Mean' or datum == 'Median' or datum == 'Mode' or datum == 'Standard Deviation' or datum == 'Variance' or datum == 'Skewness':
    #             pdf.set_font('times', 'b', 13)
    #             pdf.cell(col_width, 2*th, str(datum), border = 1, align = 'C')
    #         else:
    #             pdf.set_font('times', '', 12)
    #             pdf.cell((col_width), 2*th, str(datum), border = 1, align = 'C')
    #     pdf.ln(2*th)
    
    # pdf.ln(49)
    # pdf.write(5, 'The histogram of the inter-arrival times of the ')
    # pdf.ln(5)
    # pdf.write(5, 'dataset prior to removal of outliers is on the right.')
    # pdf.image("../plots/cleanIAThistPlot.png", x = 100, y = 105, w = 100)
    # pdf.ln(105)
    # pdf.write(5, 'The histogram of the inter-arrival times of the ')
    # pdf.ln(5)
    # pdf.write(5, 'dataset after removal of outliers using the ')
    # pdf.ln(5)
    # pdf.write(5, 'inter-quartile range method is on the right.')
    # pdf.image("../plots/inliersIAThistPlot.png", x = 100, y = 215, w = 100)
    # pdf.ln(5)
    
    
#     ''' Third Page'''
#     pdf.add_page()
#     create_heading('IAT Outliers', pdf)
#     pdf.image("../plots/donuts/outliersMetricScorePiePlot.png", x = 150, y = -5, w = 60)
#     pdf.ln(5)    
#     pdf.write(5, 'The outliers of the inter-arrival time is defined as the number of data packets which are received outside the bounds specified by the inter-quartile method.')
#     pdf.ln(10)
#     pdf.write(5, 'The Inter-Quartile Range of a dataset is defined by dividing the dataset into quartiles and selecting the middle two quartiles (50%) of values when ordered from lowest to highest.')
#     pdf.ln(10)
#     pdf.write(5, 'Quartiles are three percentiles that separate an ordered dataset into four parts. In this case, our quartiles are: 25, 50, 75')
#     pdf.ln(10)
#     pdf.write(5, 'The metric score is computed as below:')

#     pdf.image("../plots/equations/outliersMetric.png", x = 70, y = 80, w = 60)
    
#     pdf.ln(20)
#     pdf.write(5,'This score is computed on a scale from 0 to 1, with 0 being the lowest possible score, indicating that there are no data packets within the inter-quartile range, and 1 being the highest possible score indicating that all the data packets are within the inter-quartile range.')
#     pdf.ln(5)
#     pdf.write(5, 'Before applying the interquartile method, a boxplot of the dataset is given below:')
#     pdf.ln(5)
#     pdf.image("../plots/cleanBoxPlot.png", x =15, y = 115, w = WIDTH-40)
#     pdf.ln(80)
#     pdf.write(5, 'After the interquartile method is applied to remove the outliers, a boxplot of the dataset is given below:')
#     pdf.ln(5)
#     pdf.write(5, f'The value of the lower bound is: {lowerOutliers}')
#     pdf.ln(5)
#     pdf.write(5, f'The value of the upper bound is: {upperOutliers}')
#     pdf.ln(5)
#     pdf.image("../plots/inliersBoxPlot.png", x = 15, y = 215, w = WIDTH-40)
    
    
    ''' Fourth Page '''   
    # pdf.add_page()
    # pdf.ln(5)
    # create_heading('Inter-Arrival Time Outliers', pdf)
    # pdf.image("../plots/donuts/outliersMetricScorePiePlot.png", x = 150, y = -5, w = 60)
    # pdf.ln(5)
    # pdf.write(5, 'The outlier metric of the inter-arrival time is an evaluation of the number of IAT values that show a significant deviation from the expected behaviour.')
    # pdf.ln(10)
    # pdf.write(5, 'There are multiple ways to identify outliers in a dataset, and the choice of method is dependent on the independent characteristics of the dataset. In our case, we apply the modified z-score method proposed by Iglewiscz and Hoaglin.')
    # pdf.ln(5)
    # pdf.write(5,'Let the Median Absolute Deviation of the data be defined as: ')
    # pdf.ln(10)
    # pdf.image("../plots/equations/median_OutliersMetric.png", x = 60, y = 75, w = 55)
    # pdf.ln(20)
    # pdf.write(5, 'where xi is the observation for which the MAD is being computed and x is the mode of the data. We use the mode in place of the median as used by Iglewiscz and Hoaglin because we want to evaluate the deviation of the inter-arrival times from the mode, and we consider the mode to represent the expected behaviour of the dataset. Then the modified Z-score Mi is: ')
    # pdf.ln(10)
    # pdf.image("../plots/equations/modZscore_OutliersMetric.png", x = 65, y = 120, w = 45)
    # pdf.ln(20)
    # pdf.write(5, 'Here, Iglewiscz and Hoaglin suggest that observations with |Mi| > 3.5 be classified as outliers, with variations to this cut-off value depending on the distribution of x. For our purposes, we will use this value to label inter-arrival time values as outliers. The outliers for this dataset are shown in the plot below.')
    # pdf.ln(10)
    # pdf.image("../plots/outliersPlot.png", x = 15, y = 155, w = WIDTH-50)

    ''' Fifth Page '''   
    pdf.add_page()
    create_heading('Duplicate Detection', pdf)
    pdf.image("../plots/donuts/dupeMetricScorePiePlot.png", x = 150, y = -5, w = 60)
    pdf.ln(10)
    pdf.write(5, 'This metric conveys how many duplicate data points are present in the dataset.')
    pdf.ln(10)
    pdf.write(5, 'The duplicates in a dataset are identified using the timestamp and any one unique identifier for each data packet. For example: AQM Sensor ID, Vehicle ID, etc. may be used as unique identifiers for a dataset.')
    pdf.ln(5)
    pdf.write(5, 'If any unique identifier sends two data packets with the same timestamp, then one of the two data packets is counted as a duplicate. This is because it is assumed that any one device or sensor may not send two data packets with a single timestamp.')
    # pdf.ln(10)
    # pdf.write(5, 'For this dataset, the attributes chosen for deduplication are: ')
    # pdf.ln(10)
    # pdf.set_font('times', 'b', 12)
    # pdf.write(5, f'{input1}')
    # pdf.ln(5)
    # pdf.write(5, f'{input2}')
    pdf.ln(10)
    pdf.set_font('times', '', 12)
    pdf.write(5, f'Using these attributes, {dupeCount} duplicate data packets have been identified in the dataset.')
    pdf.ln(5)
    pdf.write(5, 'This metric is calculated on a score from 0 to 1, where a score of 0 indicates that all the data packets are duplicates and a score of 1 indicates that none of the data packets are duplicates.')
    pdf.ln(5)
    pdf.write(5, 'The chart below shows the number of data packets before and after deduplication on a per unique ID basis. If a unique ID is not represented in the chart, it means that there were no duplicate values received from that unique ID.')
    pdf.ln(10)
    pdf.image("../plots/dupePlotID.png", x = 20, y = 140, w = 170)
    
    ''' Sixth Page '''    
    pdf.add_page()
    create_heading('Metrics for Schema Analysis', pdf)
    pdf.ln(10)
    pdf.write(5, 'The remaining three metrics are an analysis of the metadata that is provided along with the dataset. This metadata is provided in the form of a schema, a document that delineates the different types of attributes, the data types of each attribute (integer, float, string, etc.) as well as the range of the observations under each attribute. This document also provides the mandatory attributes that the dataset must contain, as well as a list of all the expected attributes in the dataset.')
    pdf.ln(10)

    create_heading('Attribute Format Adherence', pdf)
    pdf.image("../plots/donuts/formatMetricScorePiePlot.png", x = 150, y = 50, w = 60)
    pdf.ln(10)
    pdf.write(5, 'The attribute format metric checks whether the format of the data packets being evaluated matches the format defined in the data schema. The various possible formats include number, string, float, and object.')
    pdf.ln(5)
    pdf.write(5, 'The format adherence metric is computed using the json schema validation method. The count of errors is incremented when the data type of an evaluated data packet does not match the data type specified in the data schema.')
    pdf.ln(5)
    pdf.write(5, 'A higher score for the attribute format metric indicates a relatively lower proportion of data packets that contain attributes that do not adhere to the format defined in the schema, and a lower score for the attribute format metric indicates a relatively greater proportion of data packets with incorrect attribute formats.')
    pdf.ln(10)

    create_heading('Absence of Unknown Attributes', pdf)
    pdf.image("../plots/donuts/addnlAttrMetricScorePiePlot.png", x = 150, y = 120, w = 60)
    pdf.ln(10)
    pdf.write(5, 'The unknown attributes A higher score for the attribute format metric indicates a relatively lower proportion of data packets that contain attributes that do not adhere to the format defined in the schema, and a lower score for the attribute format metric indicates a relatively greater proportion of data packets with incorrect attribute formats.metric computes the number of data packets with attributes that are present in the dataset but are not specified in the schema in any capacity.')
    pdf.ln(5)
    pdf.write(5, 'This metric is computed by validating the data against the schema. A higher score for this metric indicates a relatively lower proportion of data packets that contain attributes that are not present in the data schema and a lower score indicates a relatively greater proportion of data packets with unknown attributes.')
    pdf.ln(5)
    pdf.write(5, 'This metric represents the total number of unknown attributes in the dataset.')
    pdf.ln(10)
    
    create_heading('Adherence to Mandatory Attributes', pdf)
    pdf.image("../plots/donuts/compMetricScorePiePlot.png", x = 150, y = 195, w = 60)
    pdf.ln(10)
    pdf.write(5, 'The mandatory attributes metric checks whether the list of mandatory attributes defined in the data schema are all present in the dataset. This validation is performed for each data packet in the dataset.')
    pdf.ln(5)
    pdf.write(5, 'A higher score for the mandatory attributes metric indicates that there is a relatively greater proportion of data packets with values present for all the mandatory attributes, and a lower score for the mandatory attributes metric indicates that there is a relatively lower proportion.')
    pdf.ln(5)
    pdf.write(5, 'This metric is an indicator of the completeness of the dataset. Null values received under mandatory attributes are also included in the count of the number of missing attributes.')
    
    
    
    '''Seventh Page'''
    
    # if 'AQM' in fileName:
    #     pdf.add_page()    
        
    #     create_heading('Additional Information about the Data', pdf)
    #     pdf.ln(5)
    #     pdf.write(5, 'In this section are some useful visualizations that describe certain data statistics that can be used by the end user to determine the usability of the data. These subheadings may not explicitly fall under the umbrella of data quality and so are not counted as part of the overall score.')
        

    #     create_heading('Correlation', pdf)
    #     pdf.ln(5)
    #     pdf.write(5, "Correlation here refers to a causal relationship between different attributes found in the dataset. This relationship might be either directly or inversely proportional.")
    #     pdf.ln(5)
    #     pdf.write(5, "This relationship is shown in the heat map below, with darker colors referring to a stronger direct relationship, and lighter colors referring to a stronger inverse relationship.")
    #     pdf.image("../plots/corrPlot.jpg",x = 20, y = 80, w = 160)
    #     pdf.ln(80)


    #     create_heading('Cardinality', pdf)
    #     pdf.ln(5)
    #     pdf.write(5, 'Cardinality of a dataset is defined here as the number of unique values of in that dataset. A higher value of cardinality indicates a higher proportion of unique values.')
    #     pdf.ln(5)
    #     pdf.image("../plots/cardPlot.png",x = 35, y = 180, w = 140)
    
    pdf.output('../outputReports/' + filename, 'F')
    # pdf.output(fileName, 'F')
# if __name__ == '__main__':
#     yesterday = (datetime.today() - timedelta(days=1)).strftime("%m/%d/%y").replace("/0","/").lstrip("0")
#     # yesterday = "10/10/20" # Uncomment line for testing

# TODO//: separate function definition from function call 
create_analytics_report_schema()



#Output Report as JSON
dateTimeColumn = 'observationDateTime'
if dateTimeColumn in dfRaw.columns:
    outputParamFV = {
        "fileName": datasetName,
        "startTime": str(startTime),
        "endTime": str(endTime),
        "No. of data packets": numPackets,
        "avgDataQualityScore": avgDataQualityScore,
        # "IAT Regularity":{
        #     "overallValue": regularityMetricScore,
        #     "type": "number",
        #     "metricLabel": "IAT Regularity Metric",
        #     "metricMessage": f"For this dataset, the inter-arrival time regularity metric value is {regularityMetricScore}",
        #     "description": "This metric is rated on a scale between 0 & 1; computes the output of the equation (1 - ((No.of data packets outside the bounds)/(Total no. of data packets)). These bounds are defined by the value of alpha and the formula (mode +/- (alpha*mode)). The overall metric score is formed from an average of the three scores obtained from three values of alpha."
        # },
        # "IATOutliers":{
        #     "value": outliersMetricScore,
        #     "type": "number",
        #     "metricLabel": "IAT Outlier Metric",
        #     "metricMessage": f"For this dataset, the inter-arrival time outliers metric score is {outliersMetricScore}.",
        #     "description": "This metric is rated on a scale between 0 & 1; it is computed using the modified Z-score method and is calculated as (1-(No. of outliers/No. of data packets))"  
        # },
        # "Data Source Uptime":{
        #     "value": sensorUptimeMetricScore,
        #     "type": "number",
        #     "metricLabel": "Data Source Uptime Metric",
        #     "metricMessage": f"For this dataset, the data source uptime metric score is {sensorUptimeMetricScore}.",
        #     "description": "This metric is rated on a scale between 0 & 1; it is computed using the formula (1 - (avg. outage time per sensor/total query time))."
        # },
        "Absence of Duplicate Values":{
            "value": dupeMetricScore,
            "deduplicationAttributes": [input1, input2],
            "type": "number",
            "metricLabel": "Duplicate Value Metric",
            "metricMessage": f"For this dataset, the duplicate value metric score is: {dupeMetricScore}.",
            "description": "This metric is rated on a scale between 0 & 1; it is computed using the formula (1 - (No. of duplicate data packets/total no. of data packets)."
        },
        "Adherence to Attribute Format":{
            "value": format_adherence_metric,
            "type": "number",
            "metricLabel": "Format Adherence Metric",
            "metricMessage": "For this dataset, " + str(format_adherence_metric) + " is the format adherence",
            "description": "The metric is rated on a scale between 0 & 1; computed using the formula (1 - (no. of format validity errors/total no. of data packets))."
            },
        "Absence of Unknown Attributes":{
            "value": unknown_fields_absent_metric,
            "type": "number",
            "metricLabel": "Unknown Attributes Metric",
            "metricMessage": "For this dataset, " + str(unknown_fields_absent_metric) + " is the value of the  additional fields absent metric.",
            "description": "The metric is rated on a scale between 0 & 1; computed as (1 - r) where r is the ratio of packets with unknown attributes to the total number of packets."
            },
        "Adherence to Mandatory Attributes":{
            "value": completeness_metric,
            "type": "number",
            "metricLabel": "Completeness Metric",
            "metricMessage": "For this dataset, " + str(completeness_metric) + " is the value of the adherence to mandatory attributes metric.",
            "description": "The metric is rated on a scale between 0 & 1; It is computed as follows: For each mandatory attribute, i, compute r(i) as the ratio of packets in which attribute i is missing. Then output 1 - average(r(i)) where the average is taken over all mandatory attributes."
            }
    }
else:
    outputParamFV = {
        "fileName": datasetName,
        "No. of data packets": numPackets,
        "avgDataQualityScore": avgDataQualityScore,
        # "IAT Regularity":{
        #     "overallValue": regularityMetricScore,
        #     "type": "number",
        #     "metricLabel": "IAT Regularity Metric",
        #     "metricMessage": f"For this dataset, the inter-arrival time regularity metric value is {regularityMetricScore}",
        #     "description": "This metric is rated on a scale between 0 & 1; computes the output of the equation (1 - ((No.of data packets outside the bounds)/(Total no. of data packets)). These bounds are defined by the value of alpha and the formula (mode +/- (alpha*mode)). The overall metric score is formed from an average of the three scores obtained from three values of alpha."
        # },
        # "IATOutliers":{
        #     "value": outliersMetricScore,
        #     "type": "number",
        #     "metricLabel": "IAT Outlier Metric",
        #     "metricMessage": f"For this dataset, the inter-arrival time outliers metric score is {outliersMetricScore}.",
        #     "description": "This metric is rated on a scale between 0 & 1; it is computed using the modified Z-score method and is calculated as (1-(No. of outliers/No. of data packets))"  
        # },
        # "Data Source Uptime":{
        #     "value": sensorUptimeMetricScore,
        #     "type": "number",
        #     "metricLabel": "Data Source Uptime Metric",
        #     "metricMessage": f"For this dataset, the data source uptime metric score is {sensorUptimeMetricScore}.",
        #     "description": "This metric is rated on a scale between 0 & 1; it is computed using the formula (1 - (avg. outage time per sensor/total query time))."
        # },
        "Absence of Duplicate Values":{
            "value": dupeMetricScore,
            "deduplicationAttributes": [input1, input2],
            "type": "number",
            "metricLabel": "Duplicate Value Metric",
            "metricMessage": f"For this dataset, the duplicate value metric score is: {dupeMetricScore}.",
            "description": "This metric is rated on a scale between 0 & 1; it is computed using the formula (1 - (No. of duplicate data packets/total no. of data packets)."
        },
        "Adherence to Attribute Format":{
            "value": format_adherence_metric,
            "type": "number",
            "metricLabel": "Format Adherence Metric",
            "metricMessage": "For this dataset, " + str(format_adherence_metric) + " is the format adherence",
            "description": "The metric is rated on a scale between 0 & 1; computed using the formula (1 - (no. of format validity errors/total no. of data packets))."
            },
        "Absence of Unknown Attributes":{
            "value": unknown_fields_absent_metric,
            "type": "number",
            "metricLabel": "Unknown Attributes Metric",
            "metricMessage": "For this dataset, " + str(unknown_fields_absent_metric) + " is the value of the  additional fields absent metric.",
            "description": "The metric is rated on a scale between 0 & 1; computed as (1 - r) where r is the ratio of packets with unknown attributes to the total number of packets."
            },
        "Adherence to Mandatory Attributes":{
            "value": completeness_metric,
            "type": "number",
            "metricLabel": "Completeness Metric",
            "metricMessage": "For this dataset, " + str(completeness_metric) + " is the value of the adherence to mandatory attributes metric.",
            "description": "The metric is rated on a scale between 0 & 1; It is computed as follows: For each mandatory attribute, i, compute r(i) as the ratio of packets in which attribute i is missing. Then output 1 - average(r(i)) where the average is taken over all mandatory attributes."
            }
    }
myJSON = json.dumps(outputParamFV, indent = 4)
filename = fileNameNoExt + "_Report.json"
jsonpath = os.path.join("../outputReports/", filename)

with open(jsonpath, "w+") as jsonfile:
    jsonfile.write(myJSON)
    print("Output file successfully created.")
