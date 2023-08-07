import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib
import sys
import json
from pandas.io.json import json_normalize
import os
import math
# import panel as pn
import hvplot.pandas
from scipy.stats import norm
import seaborn as sns
from numpy import percentile
import pygal
from pygal.style import Style

#plot styles
custom_style = Style(background = 'transparent', 
                     plot_background = 'transparent', 
                     font_family = 'times')

#reading the config file
def readFile(configFile):
    with open(configFile, "r") as file:
        configDict = json.load(file)
    
    folderName = configDict['folderName']
    dataFile = '../data/' + folderName + '/' + configDict['dataFileNameJSON']    
    
    with open(dataFile, "r") as jfile:
        jsonDataDict = json.load(jfile)

    df = pd.json_normalize(jsonDataDict)
    pd.set_option('mode.chained_assignment', None)
    alpha1 = configDict['interArrivalTime']['alpha'][0]
    alpha2 = configDict['interArrivalTime']['alpha'][1]
    alpha3 = configDict['interArrivalTime']['alpha'][2]
    alpha = [alpha1, alpha2, alpha3]
    input1 = configDict['interArrivalTime']['inputFields'][0]
    input2 = configDict['interArrivalTime']['inputFields'][1]
    datasetName = configDict['datasetName']
    fileName = '../data/' + folderName + '/' + configDict['dataFileNameJSON']
    schema = '../schemas/' + configDict['schemaFileName']
    URL = configDict['URL']
    reportName = configDict['dataFileNameJSON']
    
    # print(df)
    if "Amb" in fileName:
        # df = pd.json_normalize(df['location.coordinates'])   
        df.drop('location.coordinates', axis=1, inplace = True)
        # df[['type', 'coordinates']] = df['location'].apply(','.join).str.split(expand=True)
        # df[['longitude', 'latitude']] = df['coordinates'].apply(','.join).str.split(expand = True)
    else:
        df = df
    # print(df)
    print('The loaded dataset is: ' + datasetName)
    return configDict, df, input1, input2, datasetName, fileName, URL, alpha, schema


####data preprocessing

# finding the time range of the dataset
def timeRange(dataframe):
    startTime = min(dataframe['observationDateTime'])
    startTime = pd.to_datetime(startTime)
    startTime = startTime.tz_localize(None)
    # print(type(startTime))
    # ts = pd.Timestamp('2020-03-14T15:32:52.192548651')
    # startTime.strftime('%Y-%m-%d %X')
    endTime = max(dataframe['observationDateTime'])
    endTime = pd.to_datetime(endTime)
    endTime = endTime.tz_localize(None)
    #returning Month Names and Year
    startMonth = str(startTime.month_name())[0:3]
    endMonth = str(endTime.month_name())[0:3]
    startYear = startTime.year
    endYear = endTime.year
    # endTime = pd.to_datetime(endTime, unit='ns')
    return startTime, endTime, startMonth, endMonth, startYear-2000, endYear-2000

#dropping duplicates
def dropDupes(dataframe, input1, input2):
    # dataName = dataDict['fileName']
    dfLen1 = len(dataframe)
    dfDrop = dataframe.drop_duplicates(subset = [input1, input2], inplace = False, ignore_index = True)
    dfLen2 = len(dfDrop)
    dupeCount = dfLen1 - dfLen2
    p1 = print(str(dupeCount) + ' duplicate rows have been removed.') 
    p2 = print(str(dfDrop.shape) + ' is the shape of the new dataframe.')
    dataframe = dfDrop
    return dataframe, dupeCount


#data preprocessing includes parsing datetime, sorting by id, finding timedelta, and dropping timedeltas < 0
def preProcess(df, input1, input2):
    # df['observationDateTime'] =  pd.to_datetime(df['observationDateTime'], origin = 'unix', unit = 'ms')
    df['observationDateTime'] =  pd.to_datetime(df['observationDateTime'])
    df = df[['observationDateTime', input1]]
    df.sort_values(by = [input1, input2], inplace = True, ascending = [True, True])
    df['IAT'] = df['observationDateTime'].diff().dt.total_seconds()
    df['IAT'] = df['IAT'][(df['IAT']>=0)]
    # mode = df['IAT'].mode()
    # df.sort_values(by = 'IAT', inplace = True, ascending = True)
    df = df.reset_index(drop = True) 
    return(df)

#IQR Outliers are identified and removed
def outRemove(df, dataFile, input1):
    dfInliers = df.copy(deep = True)
    dataName = dataFile
    
    #checking to see if dataset is AQM for plot labels (sensor id)
    dfInliers['idTrunc'] = dfInliers[input1].str[-4:]
    k = 1.5
    Q1 = percentile(dfInliers['IAT'].dropna(),25)
    Q3 = percentile(dfInliers['IAT'].dropna(),75)
    IQR = Q3 - Q1
    # print(Q1, Q3, IQR)
    # print('Percentiles: 25th:' + str(round(Q1)) + ', 75th:' + str(round(Q3)))
    cutOff = IQR*k
    lower, upper = round((Q1 - cutOff),3), round((Q3 + cutOff),3)
    # print(lower, upper)
    outliers = [x for x in dfInliers['IAT'] if x < lower or x > upper]
    # print(str(len(outliers)) + ' outliers have been identified within the inter quartile.')
    dfInliers.drop(dfInliers[(dfInliers['IAT'] < lower)].index, inplace = True)
    dfInliers.drop(dfInliers[(dfInliers['IAT'] > upper)].index, inplace = True)
    # print(len(df))
    dfInliers.reset_index(inplace = True, drop = True)
    # bins = np.linspace(min(df['IAT'].dropna()), max(df['IAT'].dropna()), 10)
    # df['IAT'].dropna().plot.hist(bins = bins, edgecolor = 'k', alpha = 0.5) 
    # plt.xticks(rotation = 90)
    # plt.xlabel('Inter Arrival Time [in seconds]')
    # plt.savefig('plots/'+'IAThistPlot.png', bbox_inches = 'tight', transparent = True)  
    return dfInliers, lower, upper

    
def dataStats(df):
    skewStat = int(df['IAT'].skew())
    varianceStat = int(df['IAT'].var())
    meanStat = int(df['IAT'].mean())
    modeStat = int(df['IAT'].mode()[0])
    medianStat = int(df['IAT'].median())
    stdStat = int(df['IAT'].std())
    kurtosisStat = int(df['IAT'].kurtosis())
    
    # print('The statistics for this dataset are: \n' + 
    #       'Mean = ' + str(round(meanStat, 4)) + '\n' +
    #       'Median = ' + str(round(medianStat, 4)) + '\n' +
    #       'Mode = ' + str(round(modeStat, 4)) + '\n' +
    #       'Standard Deviation = ' + str(round(stdStat, 4)) + '\n' +
    #       'Variance = ' + str(round(varianceStat, 4)) + '\n' +
    #       'Skewness = ' + str(round(skewStat, 4)) + '\n' +
    #       'Kurtosis = ' + str(round(kurtosisStat, 4)) + '\n'  
    #      )
    return meanStat, medianStat, modeStat, stdStat, varianceStat, skewStat, kurtosisStat

#####################################################################################
#####################################################################################

#DATA VISUALIZATIONS

#####################################################################################
#####################################################################################

def radarChart(regularityScore, sensorUptimeScore, dupeScore, formatScore, completeScore, addnlScore):

    custom_style = Style(
      background='transparent',
      plot_background='transparent',
      colors = ('#17C37B','#D9DFEB'),
      opacity = '0.3',
      font_family = 'times')

    radar_chart = pygal.Radar(fill = True, 
                              width = 530, 
                              height = 530, 
                              style = custom_style, 
                              show_legend = False,
                              show_title = False)
    radar_chart.x_labels = ['Regularity of InterArrival Time', 'Device Uptime','Absence of Duplicates', 'Attribute Format Adherence', 'Mandatory Attribute Adherence', 'Unknown Attribute Absence']
    radar_chart.add('Metric Scores', [regularityScore, sensorUptimeScore, dupeScore, formatScore, completeScore, addnlScore])
    radar_chart.add('Full Score', [1,1,1,1,1,1])
    radar_chart.render_to_png('../plots/radarPlot.png')
    return 


def bars(score, name):
    custom_style = Style(
    background = 'transparent',
    plot_background = 'transparent',
    colors = ('#17C37B','#C32517'),
    font_family = 'times'
    )
    
    remaining = 1 - score
    bar = pygal.HorizontalStackedBar(width = 500, 
                                height = 100, 
                                show_legend = False,
                                include_x_axis = False,
                                include_y_axis = False,
                                show_x_labels = False,
                                show_title = False,
                                style = custom_style,
                                print_values = True)
    bar.add(name, score)
    bar.add('remaining', remaining)
    bar.render_to_png('../plots/bars/' + name + 'bar.png')
    return

#plotting duplicates against preremoval on a per sensor basis
def plotDupesID(df, df1, input1):
    #checking the length of each column before and after deduplication
    sensorDupe = df.groupby(input1).size()
    sensorDupe = sensorDupe.to_frame().reset_index()
    sensorClean = df1.groupby(input1).size()
    sensorClean = sensorClean.to_frame().reset_index()
    sensorClean['valueDupe'] = sensorDupe[0]
    sensorClean.columns = [input1, 'valueClean', 'valueDupe']
    i = 0
    sensorDupePlot = pd.DataFrame(columns = [input1, 'valueDupe', 'valueClean'])
    
    #removing duplicates
    while i < len(sensorClean['valueDupe']): 
        if sensorClean['valueDupe'][i] > sensorClean['valueClean'][i]:
            sensorDupePlot = sensorDupePlot.append({input1 : sensorClean[input1][i], 'valueDupe' : sensorClean['valueDupe'][i], 'valueClean' : sensorClean['valueClean'][i]}, ignore_index = True)
        i+=1
    else:
        i+=1
        
        sensorDupePlot[input1] = sensorDupePlot[input1].str[-4:]
    #plotting the values
    bar_chart = pygal.Bar(style = custom_style, 
                          x_title = 'Truncated Sensor ID', 
                          y_title = 'No. of Data Packets',
                          legend_at_bottom = True, 
                          legend_at_bottom_columns = 2,
                          truncate_legend = -1,
                          x_label_rotation = 45,
                          print_values = False)
    bar_chart.title = 'Deduplication Result per Unique ID'
    bar_chart.x_labels = sensorDupePlot[input1]
    bar_chart.add('Pre Deduplication', sensorDupePlot['valueDupe'])
    bar_chart.add('Post Deduplication', sensorDupePlot['valueClean'])
    bar_chart.render_to_png('../plots/dupePlotID.png')
    return 

def plotDupes(dataframe, input1, input2):
    preDedupe = len(dataframe)
    dfDrop = dataframe.drop_duplicates(subset = [input1, input2], inplace = False, ignore_index = True)
    postDedupe = len(dfDrop)
    #plotting
    # print(postDedupe)
    # print(len(dfDrop))
    # print(len(dfDrop))
    bar_chart = pygal.HorizontalBar(style = custom_style, 
                          x_title = 'No. of Data Packets', 
                          legend_at_bottom = True, 
                          legend_at_bottom_columns = 2,
                          truncate_legend = -1,
                          x_label_rotation = 45,
                          print_values = True)
    bar_chart.title = 'Deduplication Result for Dataset'
    bar_chart.add('Pre Deduplication', preDedupe)
    bar_chart.add('Post Deduplication', postDedupe)
    bar_chart.render_to_png('../plots/dupePlot.png')
    # bar_chart.render()
    return

def IAThist(df):


#	def compute_histogram_bins(data, desired_bin_size):
#	    min_val = np.min(data)
#	    max_val = np.max(data)
#	    min_boundary = -1.0 * (min_val % desired_bin_size - min_val)
#	    max_boundary = max_val - max_val % desired_bin_size + desired_bin_size
#	    n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
#	    bins = np.linspace(min_boundary, max_boundary, n_bins)
#	    return bins

#	bins = compute_histogram_bins(dfClean['IAT'], 10)

    bins = np.linspace(min(df['IAT'].dropna())-0.05*min(df['IAT'].dropna()), max(df['IAT'].dropna())+0.05*max(df['IAT'].dropna()), 10)
    # bins = 100
    df['IAT'].dropna().plot.hist(bins = bins, edgecolor = 'k', alpha = 0.5, density = True) 
    # plt.xscale('log', 2)
    plt.xticks(rotation = 90)
    plt.xlabel('Inter Arrival Time [in seconds]')
    plt.ylabel('Normalized Frequency Of Occurence')
    plt.savefig('../plots/'+ df.name + 'IAThistPlot.png', bbox_inches = 'tight', transparent = True)
    # plt.show()
    plt.close()
    return

def outScatterPlot(df):
    plot = df.plot.scatter(x = 'idTrunc', y = 'IAT', rot = 90)
    return plot
    
    
def boxPlot(df, fileName, input1):
    dataName = fileName
#checking to see if dataset is AQM for plot labels (sensor id)
    if "AQM" in dataName:
        df['idTrunc'] = df[input1].str[-4:]
    else:
        df['idTrunc'] = df[input1]
# print(dfI[input1])
    figure(figsize = (15, 6))
    # df.sort_values(by='IAT', ascending=False, na_position='first')
    a = sns.boxplot(x = 'idTrunc', y = df['IAT'], data = df.sort_values(by='IAT', ascending=False, na_position='first'), color = 'seagreen')
    plt.xlabel('Truncated Sensor ID')
    plt.xticks(rotation = 90)
    plt.savefig('../plots/'+ df.name + 'BoxPlot.png', bbox_inches = 'tight', transparent = True)
    plt.close()
    return

def normalFitPlot(df):
    data = df['IAT'].dropna()
    mu, std = norm.fit(data) 
    bins = np.linspace(min(data)-0.05*min(data), max(data)+0.05*max(data), 20)
    plt.hist(data, bins=bins, density=True, alpha=0.5, edgecolor='k', linewidth = 0.5, rwidth = 1)

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin-(0.05*xmin), xmax+(0.05*xmax), 100)
    p = norm.pdf(x, mu, std)

    plt.plot(x, p, 'k', linewidth = 0.8)
    title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
    plt.title(title)
    plt.xlabel('Inter-Arrival Time')
    plt.ylabel('Frequency')
    # plt.savefig('../plots/'+ df.name + 'IATFitPlot.png', bbox_inches = 'tight', transparent = True)  
    # plt.show()
    df[df['IAT'] != 0]
    df['IAT'].plot.kde()
    plt.close()
    return mu, std

#CARDINALITY
#method to reframe list as dataframe in the case of location.coordinates in data as [latitude,longitude]
#function to check for the presence of location coordinates in dataframe
def check_col(col, df):
    if col in df:
        return True
    else:
        return False

    #splitting location coordinates into two new columns to preserve dataframe dtype if present in dataframe
    if check_col('location.coordinates',df) == True:
        # new df from the column of lists
        split_df = pd.DataFrame(df['location.coordinates'].tolist(), columns=['latitude','longitude'])
        # concat df and split_df
        df = pd.concat([df, split_df], axis=1)
        # display df
        df = df.drop('location.coordinates', axis=1)
    else:
        df = df

#function that returns the cardinality of the dataframe
def cardinal(df):
    cardinal = df.nunique()
    cardinality = cardinal.sum()/len(cardinal)
    return round(cardinality,4)


#function that plots the cardinality of the dataframe on a per column basis
def cardinal_plot(df):
    cardinal = df.nunique()
    # cardinal = cardinal.plot(kind = 'bar', figsize = (15,6))
    # plt.ylabel('Cardinality')
    # plt.xlabel('Column Names')
    # plt.xticks(rotation = 90)
    # plt.savefig('plots/'+'cardPlot.jpg', bbox_inches = 'tight', transparent = True)  
    # plt.close()
    bar_chart = pygal.Bar(style = custom_style, 
                          x_label_rotation=45, 
                          show_legend = False, 
                          print_values = False, 
                          print_values_position = 'top',
                          y_title = 'No. of unique data packets')
    bar_chart.x_labels = df.columns
    bar_chart.add('Cardinality', cardinal)
    bar_chart.render_to_png('../plots/cardPlot.png')
    plt.close()
    return

#SPARSITY or DENSITY
# Density or sparsity describe how many null values or NAN values are present in a dataset. The higher the value of the metric, the
# more dense the data is.

#function that returns the density of non-null values in the dataset
def density(df):
    nullVal = df.isnull().sum()
    count = df.count().sum()
    density = 1 - (nullVal/count).sum()
    return round(density,4)
    
#function that plots the density of the dataframe on a per column basis
def density_plot(df):
    nullVal = df.isnull().sum()
    count = df.count().sum()
    density = 1 - (nullVal/count).sum()
    # densePlot = nullVal.hvplot(kind = 'bar').opts(xrotation = 45)
    # densePlot = nullVal.plot(kind = 'bar', figsize = (15,6))
    # plt.xlabel('Column Name')
    # plt.ylabel('Density')
    # plt.xticks(rotation = 90)
    # plt.savefig('plots/'+'densPlot.jpg', bbox_inches = 'tight', transparent = True)  
    bar_chart = pygal.Bar(style = custom_style, 
                          x_label_rotation=45, 
                          show_legend = False, 
                          print_values = False, 
                          print_values_position = 'top', 
                          x_title = 'Column Name', 
                          y_title = 'No. of Non-Null Value Data Packets')
    bar_chart.title = 'Density of the Data'
    bar_chart.x_labels = df.columns
    bar_chart.add('Density', nullVal)
    bar_chart.render_to_png('../plots/densPlot.png')
    plt.close()
    return

#CORRELATION
def correlation(df):
    corr = df.corr()
    return corr

def corr_heatmap(df):
    # corr = correlation(df).hvplot.heatmap().opts(xrotation = 45)
    figure(figsize = (15, 6))
    corr = sns.heatmap(correlation(df), annot=False, cmap="YlGnBu")
    plt.savefig('../plots/'+'corrPlot.jpg', bbox_inches = 'tight', transparent = True) 
    plt.close()
    return

def piePlot(df, df1, name):
    custom_style = Style(background = 'transparent', 
                     plot_background = 'transparent',
                     colors = ('#17C37B','#D9DFEB'), 
                     value_font_size=20,
                     font_family = 'times')
    
    pie_chart = pygal.Pie(inner_radius=.4, 
                      style = custom_style,
                      show_legend = True,
                      print_values = True)
    pie_chart.title = 'Number of Duplicate Packets'
    # adding data
    pie_chart.add('Total Data Packets', len(df))
    pie_chart.add('Duplicate Data Packets', len(df)-len(df1))
    pie_chart.render_to_png('../plots/' + name + 'PiePlot.png')
    return
    
def gaugePlot(metricScore, name):
    custom_style = Style(background = 'transparent', 
                     plot_background = 'transparent',
                     colors = ('#17C37B','#C32517'), 
                     value_font_size=40)    
    gauge = pygal.SolidGauge(half_pie=True, 
                             inner_radius=0.50,
                             style=custom_style,
                             show_legend = False)
    gauge.add(name, [{'value': metricScore, 'max_value': 1}])
    gauge.render_to_png('../plots/donuts/' + name + 'PiePlot.png')
    return

def outagePlot(df, meanStat, stdStat):
    upperBound = 2*meanStat
    sensorOutage = df.loc[df['IAT'] > upperBound]
    sensorOutage.reset_index(inplace = True)
    outageTime = sensorOutage.groupby(by = 'idTrunc').sum()
    outageTime.sort_values(by = ['IAT'], inplace = True, ascending = False)
    outageTime.reset_index(inplace = True)
    outageTime.drop(['index'], axis = 1, inplace = True)
    outageTotal = outageTime['IAT'].sum()
    outageAverage = outageTime['IAT'].mean()
    
    bar_chart = pygal.Bar(show_legend = False, 
                          x_label_rotation = 45, 
                          x_title = 'Truncated Device ID',  
                          y_title = 'Total Outage Time in Minutes',
                          style = custom_style)
    bar_chart.title = 'Outage Time per Device'
    bar_chart.add('Outage Time per Device', outageTime['IAT']/60)
    bar_chart.x_labels = outageTime['idTrunc']
    bar_chart.render_to_png('../plots/sensorOutagePlot.png')
    
    return outageAverage

#####################################################################################
#####################################################################################

#METRIC CALCULATIONS

#####################################################################################
#####################################################################################

#Here we are calculating the penalty on the outliers for each interarrival time
def iatMetricOutliers(df):
    Q1 = percentile(df['IAT'].dropna(), 25)
    Q3 = percentile(df['IAT'].dropna(), 75)
    k = 1.5
    IQR = Q3 - Q1
    cutOff = IQR*k
    lower, upper = Q1 - cutOff, Q3 + cutOff
    outliers = [x for x in df['IAT'] if x < lower or x > upper] 
    outlierNumber = len(outliers)
    totalDataPackets = len(df)
    iatMetricScore = 1 - (outlierNumber/totalDataPackets)
    iatMetricPercent = round(iatMetricScore*100, 3)
    # print(str(iatMetricPercent) + '% of the data packets lie within the range of the lower and upper bounds defined by the Inter-Quartile Range and the choice of alpha.')
    return round(iatMetricScore,3)

#Here we are calculating the regularity of the interarrival times
def iatMetricRegularity(df, alpha): 
    #user defined input alpha
    # alpha1 = alpha[0]
    # alpha2 = alpha[1]
    # alpha3 = alpha[2]
    # alpha = [alpha1, alpha2, alpha3]
    mode = df['IAT'].mode()[0]
    
    #creating two arrays with upper and lower bounds for each alpha in mode +/- alpha*mode
    i = 0
    floor = []
    ceil = []
    while i < len(alpha):
        floor.append(mode - alpha[i]*mode)
        ceil.append(mode + alpha[i]*mode)
        # print(floor[i])
        # print(ceil[i])
        i+=1
    
    #calculating number of values that are outside the desired range for each value of alpha
    outliersA1 = [x for x in df['IAT'] if x < floor[0] or x > ceil[0]]
    outliersA2 = [x for x in df['IAT'] if x < floor[1] or x > ceil[1]]
    outliersA3 = [x for x in df['IAT'] if x < floor[2] or x > ceil[2]]
    
    numOutliersA1 = len(outliersA1)
    numOutliersA2 = len(outliersA2)
    numOutliersA3 = len(outliersA3)
    numOutliers = [numOutliersA1, numOutliersA2, numOutliersA3]
    # print(numOutliers)
    
    #calculating the value of the regularity metric
    regularityMetricScore = [0,0,0]
    i = 0
    while i < len(alpha):
        regularityMetricScore[i] = round(1 - (numOutliers[i]/len(df)), 6)
        i += 1
    
    #taking the average of the three values for overall score
    regularityMetricScoreAvg = sum(regularityMetricScore)/len(regularityMetricScore)
    return round(regularityMetricScoreAvg, 3), regularityMetricScore, floor, ceil


def dupeMetric(df, input1, input2):
    # dupeCount = len(df)-len(df.drop_duplicates(subset = [data_dict['duplicateDetection']["inputFields"][0], data_dict['duplicateDetection']['inputFields'][1]]))
    dupeCount = len(df) - len(df.drop_duplicates(subset = [input1, input2]))
    totalDataPackets = len(df)
    dupeMetricScore = 1 - dupeCount/totalDataPackets
    dupeMetricPercent = round(dupeMetricScore*100, 4)
    # print(str(dupeMetricPercent) + '% of the data packets are non duplicates, as defined by the parameters ' + str(input1) ' &' + str(input2))
    return round(dupeMetricScore,3)

def outageMetric(dfClean, dfRaw, meanStat, input1):
    #upper bound to define  sensor outage
    upperBound = 2*meanStat
    #creating a dataframe with IAT values greater than upperbound
    sensorOutage = dfClean.loc[dfClean['IAT'] > upperBound]
    #finding the sum of the outages for each sensor
    sensorOutage = sensorOutage.groupby([input1])[['IAT']].agg('sum').reset_index()    
    #calculating average of the outage for all the sensors
    avgOutageTime = sensorOutage['IAT'].mean()
    
    #finding total query time
    dfRaw['observationDateTime'] = pd.to_datetime(dfRaw['observationDateTime'])
    startTime = min(dfRaw['observationDateTime'])
    startTime = pd.to_datetime(startTime, unit='s') 
    endTime = max(dfRaw['observationDateTime'])
    endTime = pd.to_datetime(endTime, unit = 's')
    queryTime = ((endTime - startTime).total_seconds())
    outageMetric = round((1 - avgOutageTime/(queryTime)),3)
    return outageMetric
