import PreProcessing as pp

configFile = '../config/'+input('Enter the name of the configuration file: ')

dfRaw, input1, input2, datasetName, fileName, URL, alpha, schema = pp.readFile(configFile)


# In[5]:


print(fileName)
print(datasetName)
# print(os.path.splitext(os.path.basename(fileName))[0])


# In[6]:


startTime, endTime, startMonth, endMonth, startYear, endYear = pp.timeRange(dfRaw)
numPackets = dfRaw.shape[0]


# ### Validating Data against Schema

# In[21]:


import ijson
import jsonschema
import fastjsonschema
import json
import sys
import requests
import re
import logging
import os

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

#Main program
# if len(sys.argv) < 2:
#     print('###########################################################################')
#     print("Not enough arguments")
#     print("Usage: python3 validate_format <ConfigFilePath>")
#     print('###########################################################################')
#     sys.exit()

#logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

schemaFile = schema
dataFile = fileName
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

num_samples, err_count, err_data_arr, add_err_count, req_err_cnt = validate_data_with_schema(dataFile, schema)

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
num_samples, total_missing_count = validate_requiredFields(dataFile, req)

logging.debug("Total missing count: " + str(total_missing_count))


completeness_metric = 1 - total_missing_count/(num_samples*len(req))

logging.debug('###########################################################################')
logging.debug('##### Total Missing Fields Count for Required fields #######')
logging.debug("Total samples: " + str(num_samples))
logging.debug("Attribute_Completeness_Metric: "+str(completeness_metric))
logging.debug('###########################################################################')


# ### Running data preprocessing functions

# In[7]:


#dropping duplicates
dfDropped, dupeCount = pp.dropDupes(dfRaw, input1, input2)
# print(dfDropped['observationDateTime'])
#cleaning dataframe
dfClean = pp.preProcess(dfDropped, input1, input2)
# print(dfClean['IAT'].tail())
# from numpy import percentile
# print(percentile(dfClean['IAT'].dropna(),100))
# print(dfClean['IAT'].head())


#removing outliers
dfInliers, lowerOutliers, upperOutliers = pp.outRemove(dfClean, datasetName, input1)
# print(dfInliers, lowerOutliers, upperOutliers)

#datastats before/after removing outliers
meanStatOut, medianStatOut, modeStatOut, stdStatOut, varianceStatOut, skewStatOut, kurtosisStatOut = pp.dataStats(dfClean)
meanStatIn, medianStatIn, modeStatIn, stdStatIn, varianceStatIn, skewStatIn, kurtosisStatIn = pp.dataStats(dfInliers)


# In[8]:


# print(dfClean['IAT'])
# dfClean.to_csv('dfCleantest.csv')


# In[13]:


# dfInliers, lowerOutliers, upperOutliers = pp.outRemove(dfClean, datasetName, input1)
# print(lowerOutliers, upperOutliers)
# dfInliers.to_csv('dfInlierstest.csv')
# print(datasetName)


# In[38]:


#running functions that are used to calcalate the metric scores
regularityMetricScore, regularityValues, lowerRegularity, upperRegularity = pp.iatMetricRegularity(dfClean, alpha)
# outliersMetricScore = pp.iatMetricOutliers(dfClean)
sensorUptimeMetricScore = pp.outageMetric(dfClean, dfRaw, meanStatIn, input1)
dupeMetricScore = pp.dupeMetric(dfRaw, input1, input2)
compMetricScore = round(completeness_metric, 3)
formatMetricScore = round(format_adherence_metric, 3)
addnlAttrMetricScore = round(unknown_fields_absent_metric, 3)
avgDataQualityScore = round((regularityMetricScore + sensorUptimeMetricScore + dupeMetricScore + compMetricScore + formatMetricScore + addnlAttrMetricScore)/6, 3)
avgDataQualityPercent = round(avgDataQualityScore*100,3)

logging.info('################## Final Metrics #########################################')
logging.info('#')
logging.info("Regularity of Inter-Arrival Time Metric: " + str(regularityMetricScore))
logging.info('#')
logging.info("Device Uptime Metric: " + str(sensorUptimeMetricScore))
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


# In[16]:


#naming dataframes for plot file naming
dfRaw.name = 'raw'
dfDropped.name = 'dropped'
dfClean.name = 'clean'
dfInliers.name = 'inliers'


# ### Generating visualizations for the PDF in order of appearance in the report

# In[37]:


#DQ overview horizontal bars
pp.bars(regularityMetricScore, 'regularity')
# pp.bars(outliersMetricScore, 'outliers')
pp.bars(dupeMetricScore, 'dupe')
pp.bars(compMetricScore, 'comp')
pp.bars(formatMetricScore, 'format')
pp.bars(addnlAttrMetricScore, 'addnl')
pp.bars(sensorUptimeMetricScore, 'sensorUptime')


# In[38]:


#half pie charts
pp.gaugePlot(regularityMetricScore, 'regularityMetricScore')
# pp.gaugePlot(outliersMetricScore, 'outliersMetricScore')
pp.gaugePlot(dupeMetricScore, 'dupeMetricScore')
pp.gaugePlot(compMetricScore, 'compMetricScore')
pp.gaugePlot(formatMetricScore, 'formatMetricScore')
pp.gaugePlot(addnlAttrMetricScore, 'addnlAttrMetricScore')
pp.gaugePlot(sensorUptimeMetricScore, 'sensorUptimeMetricScore')
#radar chart
pp.radarChart(regularityMetricScore, 
              # outliersMetricScore, 
              sensorUptimeMetricScore,
              dupeMetricScore, 
              compMetricScore, 
              formatMetricScore, 
              addnlAttrMetricScore
              )


# #interarrival time boxplots and histogram
pp.IAThist(dfClean)
pp.boxPlot(dfClean, fileName, input1)

#without outliers boxplots and histograms
pp.IAThist(dfInliers)
pp.boxPlot(dfInliers, fileName, input1)


# In[40]:


pp.normalFitPlot(dfClean)


# In[44]:


#duplicates bar chart
pp.plotDupesID(dfRaw, dfDropped, input1)
# pp.plotDupes(dfRaw)
pp.piePlot(dfRaw, dfClean, 'dupe')


# In[45]:


#data statistics plots
#correlation
if 'AQM' in fileName:
	pp.corr_heatmap(dfDropped)

#skewness and kurtosis calculated for inlier values only
muFitIn, stdFitIn = pp.normalFitPlot(dfInliers)

#skewness and kurtosis calculated for outlier values included
muFitOut, stdFitOut = pp.normalFitPlot(dfClean)

#density/sparseness
pp.density_plot(dfDropped)

#cardinality
if 'AQM' in fileName:
	pp.cardinal_plot(dfDropped)


# In[46]:


outageAverage = pp.outagePlot(dfClean, meanStatOut, stdStatOut)


# In[91]:


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
    pdf.set_text_color(r = 0, g = 0, b = 0)
    pdf.ln(7)
    # pdf.set_x(120)
    pdf.cell(15, 5, f'Number of Data Packets: {numPackets}    |    Start Time: {startTime}    |   End Time: {endTime}')
    # pdf.ln(5)
    # pdf.set_x(120)
    # pdf.write(5, f'End Time: {endTime}')
    # pdf.ln(5)
    # pdf.set_x(120)
    # pdf.write(5, f'Number of Data Packets: {numPackets}')
    pdf.line(11, 41, 200, 41)


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
def create_analytics_report(filename=f"{fileNameNoExt}_DQReport.pdf"):
    
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
            ['Regularity of Inter-Arrival Time',f'{regularityMetricScore}', ''],
            # ['Outlier Presence in Inter-Arrival Time', f'{outliersMetricScore}', ''],
            ['Device Uptime', f'{sensorUptimeMetricScore}',''],
            ['Absence of Duplicate Values',f'{dupeMetricScore}', ''],
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
    pdf.image("../plots/bars/regularitybar.png", 107, 70, 95)
    # pdf.image("../plots/bars/outliersbar.png", 107, 88, 95)
    pdf.image("../plots/bars/sensorUptimebar.png", 107, 87, 95)
    pdf.image("../plots/bars/dupebar.png", 107, 104, 95)
    pdf.image("../plots/bars/formatbar.png", 107, 121, 95)
    pdf.image("../plots/bars/addnlbar.png", 107, 138, 95)
    pdf.image("../plots/bars/compbar.png", 107, 155, 95)
    
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
    pdf.add_page()
    # pdf.image("plots/pretty/IUDXlogo.png", 0, 0, w = 60)
    pdf.ln(5)
    create_heading('Regularity of Inter-Arrival Time ', pdf)
    pdf.ln(5)
    pdf.write(5, 'Inter-arrival time is defined as the time elapsed after the receipt of a data packet and until the receipt of the next packet. For sensor data, this is an important factor to evaluate as sensors are often configured to send data at specific time intervals.')
    pdf.ln(5)
    pdf.write(5, 'In this section, we will be analysing the regularity and anomalous values of the inter-arrival times of this dataset.')
    # create_heading('IAT Regularity', pdf)
    pdf.image("../plots/donuts/regularityMetricScorePiePlot.png", x = 150, y = -5, w = 60)
    pdf.ln(10)    
    pdf.write(5, 'The regularity metric of the inter-arrival time conveys how uniform this time interval is for a dataset in relation to the expected behaviour.')
    pdf.ln(5)
    pdf.write(5, 'Considering the mode of the inter-arrival times to be the expected value (as per the specification), this metric measures the proximity of the spread of the normal distribution to the mode. The spread of the inter-arrival time values from the mode is computed using the formula:')
    pdf.image('../plots/equations/modeAlpha.png', x = 80, y = 95, w = 40)
    pdf.ln(20)
    pdf.write(5, f'where alpha is a constant from 0 to 1. In this case, 3 values have been considered: {alpha[0]}, {alpha[1]}, {alpha[2]}.')
    pdf.ln(10)
    pdf.write(5, 'Considering the minimum and maximum values of this formula to be the lower and upper bounds, we compute the number of inter-arrival time values outside these bounds and divide by the total number of data packets using this formula:')
    pdf.image('../plots/equations/regularityMetric.png', x = 60, y = 130, w = 85)
    pdf.ln(20)
    pdf.write(5, 'This value is computed for each alpha, and then averaged to give the overall metric score. The score is on a scale from 0 to 1, where 1 indicates the highest possible proximity to the mode and 0 indicates the opposite. The average of these three scores are taken to form the overall metric score.')
    pdf.ln(10)
    #creating table for alpha values 
    
    pdf.image('../plots/equations/modeAlphaNeg.png', x = 64, y = 164, w = 30)
    pdf.image('../plots/equations/modeAlphaPos.png', x = 112, y = 165, w = 30)
    
    dataAlpha = [
                ['Alpha', '', '', 'Regularity Score'],
                [f'{alpha[0]}', f'{lowerRegularity[0]}', f'{upperRegularity[0]}', f'{regularityValues[0]}'],
                [f'{alpha[1]}', f'{lowerRegularity[1]}', f'{upperRegularity[1]}', f'{regularityValues[1]}'],
                [f'{alpha[2]}', f'{lowerRegularity[2]}', f'{upperRegularity[2]}', f'{regularityValues[2]}']
                ]

    # Text height is the same as current font size
    # Effective page width, or just epw
    epw = pdf.w - 2*pdf.l_margin

    # Set column width to 1/3 of effective page width to distribute content 
    # evenly across table and page
    col_width = epw/4
    th = pdf.font_size
    
    #logic included to bold only titles
    for row in dataAlpha:
        for index, datum in enumerate(row):
            if datum == 'Alpha' or datum == 'Regularity Score':
                pdf.set_font('times', 'b', 13)
                pdf.cell(col_width, 2*th, str(datum), border = 1, align = 'C')
            else: 
                pdf.set_font('times', '', 12)
                pdf.cell(col_width, 2*th, str(datum), border = 1, align = 'C')
        pdf.ln(2*th)
    pdf.ln(10)
    pdf.write(5, 'A high score for the inter-arrival time metric means that the data packets are received at regular intervals which is important for time-critical applications.')
    # pdf.ln(100)
    # pdf.write(5, 'The table below shows a comparison of the statistics of the inter-arrival times of the dataset before and after outlier treatment using the Inter-Quartile Range method.')
    # pdf.ln(20)
    
#     #creating a table for the inter arrival time statistics
#     dataStats = [
#                 ['','Before Outlier Removal', 'After Removal of Outliers'],
#                 ['Mean', f'{meanStatOut}', f'{meanStatIn}'],
#                 ['Median', f'{medianStatOut}', f'{medianStatIn}'],
#                 ['Mode', f'{modeStatOut}', f'{modeStatIn}'],
#                 ['Standard Deviation', f'{stdStatOut}', f'{stdStatIn}'],
#                 ['Variance', f'{varianceStatOut}', f'{varianceStatIn}'],
#                 ['Skewness', f'{skewStatOut}', f'{skewStatIn}']
#                 ]
    
#     col_width = epw/3
#     #logic included to bold only titles
#     for row in dataStats:
#         for index, datum in enumerate(row):
#             if datum == 'Before Outlier Removal' or datum == 'After Removal of Outliers' or datum == 'Mean' or datum == 'Median' or datum == 'Mode' or datum == 'Standard Deviation' or datum == 'Variance' or datum == 'Skewness':
#                 pdf.set_font('times', 'b', 13)
#                 pdf.cell(col_width, 2*th, str(datum), border = 1, align = 'C')
#             else:
#                 pdf.set_font('times', '', 12)
#                 pdf.cell((col_width), 2*th, str(datum), border = 1, align = 'C')
#         pdf.ln(2*th)
    
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
    pdf.add_page()
    pdf.ln(5)
    create_heading('Device Uptime', pdf)
    pdf.image("../plots/donuts/sensorUptimeMetricScorePiePlot.png", x = 150, y = -5, w = 60)
    pdf.ln(5)
    pdf.write(5, 'Device uptime is defined as the duration in which the device is actively sending data packets at the expected time intervals.')
    pdf.ln(10)
    pdf.write(5, 'This metric is calculated by performing an analysis of the inter-arrival time of the devices. Each value of the inter-arrival time that is greater than twice the mean is selected and sorted by device. These values are then summed for each device and an overall average is taken. This overall average value is then divided by the total query time of the dataset.')
    pdf.ln(5)
    pdf.write(5,'Total query time is the time for which the dataset is queried, i.e. the difference between the timestamps of the first and last data packets in the dataset.')
    pdf.ln(10)
    pdf.write(5, 'The metric score is computed as below:')
    pdf.image("../plots/equations/sensorUptimeMetric.png", x = 60, y = 90, w = 75)
    pdf.ln(20)
    pdf.write(5, 'Assuming that a high value for the inter-arrival time means that the device is not sending data packets at the expected intervals and is assumed to be "down". Device uptime can be understood as the time during which the device is not undergoing an outage and is functioning as expected.')
    pdf.ln(10)
    pdf.write(5, 'The metric is calculated on a scale from 0 to 1, with 0 being the lowest score indicating that there is a high degree of device outage in the dataset, and 1 being the highest score indicating that there are no inter-arrival times greater than twice the mean.')
    
    # pdf.add_page()
    pdf.ln(10)
    pdf.write(5, 'The chart below shows the device downtime - time when the device is not actively sending data packets at the expected intervals on a "per device" basis. This chart only shows the downtime if the device has experienced downtime that is greater than twice the mean of the inter-arrival time.')
    pdf.image("../plots/sensorOutagePlot.png", x = 20, y = 160, w = WIDTH - 60)

    ''' Fifth Page '''   
    pdf.add_page()
    create_heading('Duplicate Detection', pdf)
    pdf.image("../plots/donuts/dupeMetricScorePiePlot.png", x = 150, y = -5, w = 60)
    pdf.ln(10)
    pdf.write(5, 'This metric conveys how many duplicate data points are present in the dataset.')
    pdf.ln(10)
    pdf.write(5, 'The duplicates in a dataset are identified using the timestamp and any one unique identifier for each data packet. For example: AQM Sensor ID, Vehicle ID, etc. may be used as unique identifiers for a dataset.')
    pdf.ln(5)
    pdf.write(5, 'If any unique identifier sends two data packets with the same timestamp, then one of the two data packets is counted as a duplicate. This is because it is assumed that any one sensor may not send two data packets with a single timestamp.')
    pdf.ln(10)
    pdf.write(5, 'For this dataset, the attributes chosen for deduplication are: ')
    pdf.ln(10)
    pdf.set_font('times', 'b', 12)
    pdf.write(5, f'{input1}')
    pdf.ln(5)
    pdf.write(5, f'{input2}')
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
    create_heading('Attribute Format Adherence', pdf)
    pdf.image("../plots/donuts/formatMetricScorePiePlot.png", x = 150, y = -5, w = 60)
    pdf.ln(10)
    pdf.write(5, 'This metric assesses the level of adherence of the data to its expected format as defined in the data schema.')
    pdf.ln(5)
    pdf.write(5, 'It is quantified by taking the ratio of packets adhering to the expected schemas to the total number of data packets.')
    pdf.ln(10)
    pdf.write(5, 'Represents the completeness of the attributes of the dataset.')
    pdf.ln(40)
    
    create_heading('Absence of Unknown Attributes', pdf)
    pdf.image("../plots/donuts/addnlAttrMetricScorePiePlot.png", x = 150, y = 80, w = 60)
    pdf.ln(10)
    pdf.write(5, 'This metric checks whether there are any additional attributes present in the dataset apart from the list of required attributes.')
    pdf.ln(10)
    pdf.write(5, 'This metric is computed as (1 - r) where r is the ratio of packets with unknown fields (fields that are not present in the list of mandatory attributes) to the total number of packets.')
    pdf.ln(10)
    # pdf.image("../plots/equations/unknownAttributesMetric.png")
    # pdf.ln(10)
    pdf.write(5, 'This metric represents the total number of unknown attributes in the dataset.')
    pdf.ln(20)
    
    create_heading('Adherence to Mandatory Attributes', pdf)
    pdf.image("../plots/donuts/compMetricScorePiePlot.png", x = 150, y = 155, w = 60)
    pdf.ln(10)
    pdf.write(5, 'This metric checks whether all the required attributes defined in the schema are present in the dataset.')
    pdf.ln(10)
    pdf.write(5, 'It is computed as follows: For each mandatory attribute, i, compute r(i) as the ratio of packets in which attribute i is missing. Then output 1 - average(r(i)) where the average is taken over all mandatory attributes.')
    pdf.ln(10)
    pdf.write(5, 'The metric is computed on a scale from 0 to 1, where a score of 0 indicates that all the data packets in the dataset are missing the required attributes, and 1 indicating that all the data packets are adherent to the list of required attributes. The metric represents the completeness of the attributes of the dataset.')
    
    
    
    '''Seventh Page'''
    
    if 'AQM' in fileName:
	    pdf.add_page()    
	    
	    create_heading('Additional Information about the Data', pdf)
	    pdf.ln(5)
	    pdf.write(5, 'In this section are some useful visualizations that describe certain data statistics that can be used by the end user to determine the usability of the data. These subheadings may not explicitly fall under the umbrella of data quality and so are not counted as part of the overall score.')
	    

	    create_heading('Correlation', pdf)
	    pdf.ln(5)
	    pdf.write(5, "Correlation here refers to a causal relationship between different attributes found in the dataset. This relationship might be either directly or inversely proportional.")
	    pdf.ln(5)
	    pdf.write(5, "This relationship is shown in the heat map below, with darker colors referring to a stronger direct relationship, and lighter colors referring to a stronger inverse relationship.")
	    pdf.image("../plots/corrPlot.jpg",x = 20, y = 80, w = 160)
	    pdf.ln(80)


	    create_heading('Cardinality', pdf)
	    pdf.ln(5)
	    pdf.write(5, 'Cardinality of a dataset is defined here as the number of unique values of in that dataset. A higher value of cardinality indicates a higher proportion of unique values.')
	    pdf.ln(5)
	    pdf.image("../plots/cardPlot.png",x = 35, y = 180, w = 140)
    
    pdf.output('../outputReport/' + filename, 'F')
    # pdf.output(fileName, 'F')
# if __name__ == '__main__':
#     yesterday = (datetime.today() - timedelta(days=1)).strftime("%m/%d/%y").replace("/0","/").lstrip("0")
#     # yesterday = "10/10/20" # Uncomment line for testing

create_analytics_report()


# In[63]:


#Output Report as JSON

outputParamFV = {
    "fileName": datasetName,
    "startTime": str(startTime),
    "endTime": str(endTime),
    "No. of data packets": numPackets,
    "avgDataQualityScore": avgDataQualityScore,
    "IAT Regularity":{
        "value": [regularityValues[0], regularityValues[1], regularityValues[2]],
        "valueAlpha": [alpha[0], alpha[1], alpha[2]],
        "overallValue": regularityMetricScore,
        "type": "number",
        "metricLabel": "IAT Regularity Metric",
        "metricMessage": f"For this dataset, the inter-arrival time regularity metric values are: {regularityValues[0]}, {regularityValues[1]},  and {regularityValues[2]} for the corresponding alpha values of {alpha[0]}, {alpha[1]}, and {alpha[2]}. The overall score for this metric is {regularityMetricScore}",
        "description": "This metric is rated on a scale between 0 & 1; computes the output of the equation (1 - ((No.of data packets outside the bounds)/(Total no. of data packets)). These bounds are defined by the value of alpha and the formula (mode +/- (alpha*mode)). The overall metric score is formed from an average of the three scores obtained from three values of alpha."
    },
    # "IATOutliers":{
    #     "value": outliersMetricScore,
    #     "type": "number",
    #     "metricLabel": "IAT Outlier Metric",
    #     "metricMessage": f"For this dataset, the inter-arrival time outliers metric score is {outliersMetricScore}.",
    #     "description": "This metric is rated on a scale between 0 & 1; it is computed using the inter-quartile range method and is calculated as (1-(No. of outliers/No. of data packets))"  
    # },
    "Device Uptime":{
        "value": sensorUptimeMetricScore,
        "type": "number",
        "metricLabel": "Device Uptime Metric",
        "metricMessage": f"For this dataset, the device uptime metric score is {sensorUptimeMetricScore}.",
        "description": "This metric is rated on a scale between 0 & 1; it is computed using the formula (1 - (avg. outage time per device/total query time))."
    },
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
jsonpath = os.path.join("../outputReport/", filename)

with open(jsonpath, "w+") as jsonfile:
    jsonfile.write(myJSON)
    print("Output file successfully created.")
