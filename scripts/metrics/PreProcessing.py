import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import json
from pandas import json_normalize
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
    """
    Reads configuration file and data file.

    Reads configuration file and extracts configuration parameters.
    Reads data file and loads data into a pandas DataFrame.

    Parameters:
        configFile (str): Path to configuration file

    Returns:
        configDict (dict): Configuration parameters
        dfRaw (pd.DataFrame): Raw data
        input1 (str): First attribute for deduplication
        input2 (str): Second attribute for deduplication
        datasetName (str): Name of the dataset
        fileName (str): Name of the data file
        URL (str): URL link to the dataset
        alpha (list): List of alpha values for IAT regularity metric
        schema (dict or None): Schema for data validation (if schema validation is enabled)

    """

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
        df.drop('location.coordinates', axis=1, inplace = True)
    else:
        df = df
    print('The loaded dataset is: ' + datasetName)
    return configDict, df, input1, input2, datasetName, fileName, URL, alpha, schema

# finding the time range of the dataset
def timeRange(dataframe):
    """
    Finds the time range of the dataset.

    Args:
        dataframe (pd.DataFrame): DataFrame with 'observationDateTime' column

    Returns:
        startTime (pd.Timestamp): Start time of the dataset
        endTime (pd.Timestamp): End time of the dataset
        startMonth (str): Start month (e.g. Jan)
        endMonth (str): End month (e.g. Dec)
        startYear (int): Start year (e.g. 2020)
        endYear (int): End year (e.g. 2021)
    """
    startTime = min(dataframe['observationDateTime'])
    startTime = pd.to_datetime(startTime)
    startTime = startTime.tz_localize(None)
    endTime = max(dataframe['observationDateTime'])
    endTime = pd.to_datetime(endTime)
    endTime = endTime.tz_localize(None)
    startMonth = str(startTime.month_name())[0:3]
    endMonth = str(endTime.month_name())[0:3]
    startYear = startTime.year
    endYear = endTime.year
    return startTime, endTime, startMonth, endMonth, startYear-2000, endYear-2000

#dropping duplicates
def dropDupes(dataframe, input1, input2):
    """
    Drops duplicate rows from a DataFrame based on two columns.

    This function takes a DataFrame and two column names as input. It then
    identifies and removes duplicate rows from the DataFrame based on the two
    columns. The function returns the updated DataFrame and the number of
    duplicate rows that were removed.

    Parameters:
        dataframe (pd.DataFrame): DataFrame to drop duplicates from
        input1 (str): First column name to check for duplicates
        input2 (str): Second column name to check for duplicates

    Returns:
        dfDrop (pd.DataFrame): DataFrame with duplicates removed
        dupeCount (int): Number of duplicate rows that were removed
    """
    dfLen1 = len(dataframe)
    dfDrop = dataframe.drop_duplicates(subset = [input1, input2], inplace = False, ignore_index = True)
    dfLen2 = len(dfDrop)
    dupeCount = dfLen1 - dfLen2
    p1 = print(str(dupeCount) + ' duplicate rows have been removed.') 
    p2 = print(str(dfDrop.shape) + ' is the shape of the new dataframe.')
    dataframe = dfDrop
    return dataframe, dupeCount

#IQR Outliers are identified and removed
def outRemove(df, dataFile, input1):
    """
    Identifies and removes outliers from a DataFrame based on IQR threshold.

    Parameters:
        df (pd.DataFrame): DataFrame with IAT column
        dataFile (str): Name of the data file
        input1 (str): First attribute for deduplication

    Returns:
        dfInliers (pd.DataFrame): DataFrame with outliers removed
        lower (float): Lower bound of the IQR range
        upper (float): Upper bound of the IQR range
    """
    dfInliers = df.copy(deep = True)
    dataName = dataFile
    
    #checking to see if dataset is AQM for plot labels (sensor id)
    dfInliers['idTrunc'] = dfInliers[input1].str[-4:]
    k = 1.5
    Q1 = percentile(dfInliers['IAT'].dropna(),25)
    Q3 = percentile(dfInliers['IAT'].dropna(),75)
    IQR = Q3 - Q1
    cutOff = IQR*k
    lower, upper = round((Q1 - cutOff),3), round((Q3 + cutOff),3)
    outliers = [x for x in dfInliers['IAT'] if x < lower or x > upper]
    dfInliers.drop(dfInliers[(dfInliers['IAT'] < lower)].index, inplace = True)
    dfInliers.drop(dfInliers[(dfInliers['IAT'] > upper)].index, inplace = True)
    dfInliers.reset_index(inplace = True, drop = True)
    return dfInliers, lower, upper

    
def dataStats(df):
    """
    Compute summary statistics for IAT column in DataFrame.

    Args:
        df: DataFrame with IAT column

    Returns:
        tuple of 7 integers: mean, median, mode, standard deviation, variance, skewness, kurtosis
    """
    
    skewStat = int(df['IAT'].skew())
    varianceStat = int(df['IAT'].var())
    meanStat = int(df['IAT'].mean())
    modeStat = int(df['IAT'].mode()[0])
    medianStat = int(df['IAT'].median())
    stdStat = int(df['IAT'].std())
    kurtosisStat = int(df['IAT'].kurtosis())

    return meanStat, medianStat, modeStat, stdStat, varianceStat, skewStat, kurtosisStat

#Data visualizations
def radarChart(regularityScore, outliersScore, dupeScore, formatScore, completeScore, addnlScore):

    """
    Creates a radar chart to visualize the metric scores.

    Args:
        regularityScore (float): Regularity of InterArrival Time metric score
        outliersScore (float): Outliers of Inter-Arrival Time metric score
        dupeScore (float): Absence of Duplicates metric score
        formatScore (float): Attribute Format Adherence metric score
        completeScore (float): Mandatory Attribute Adherence metric score
        addnlScore (float): Unknown Attribute Absence metric score

    Returns:
        None
    """

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
    radar_chart.x_labels = ['Regularity of InterArrival Time', 'Outliers of Inter-Arrival Time','Absence of Duplicates', 'Attribute Format Adherence', 'Mandatory Attribute Adherence', 'Unknown Attribute Absence']
    radar_chart.add('Metric Scores', [regularityScore, outliersScore, dupeScore, formatScore, completeScore, addnlScore])
    radar_chart.add('Full Score', [1,1,1,1,1,1])
    radar_chart.render_to_png('../plots/radarPlot.png')
    return 


def bars(score, name):
    """
    Creates a horizontal stacked bar chart to visualize a single metric score.

    Args:
        score (float): The metric score to visualize
        name (str): The name of the metric

    Returns:
        None
    """
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
    """
    Creates a bar chart to visualize the effect of deduplication on data packets per sensor.

    Args:
        df (pd.DataFrame): Original DataFrame before deduplication.
        df1 (pd.DataFrame): DataFrame after deduplication.
        input1 (str): Column name representing sensor IDs.

    Returns:
        None
    """

    sensorDupe = df.groupby(input1).size()
    sensorDupe = sensorDupe.to_frame().reset_index()
    sensorClean = df1.groupby(input1).size()
    sensorClean = sensorClean.to_frame().reset_index()
    sensorClean['valueDupe'] = sensorDupe[0]
    sensorClean.columns = [input1, 'valueClean', 'valueDupe']
    index_names = sensorClean[ (sensorClean['valueClean'] == sensorClean['valueDupe'])].index
    sensorClean.drop(index_names, inplace = True)
    bar_chart = pygal.Bar(style = custom_style, 
                          x_title = 'Truncated Sensor ID', 
                          y_title = 'No. of Data Packets',
                          legend_at_bottom = True, 
                          legend_at_bottom_columns = 2,
                          truncate_legend = -1,
                          x_label_rotation = 45,
                          print_values = False)
    bar_chart.title = 'Deduplication Result per Unique ID'
    bar_chart.x_labels = sensorClean[input1]
    bar_chart.add('Pre Deduplication', sensorClean['valueDupe'])
    bar_chart.add('Post Deduplication', sensorClean['valueClean'])
    bar_chart.render_to_png('../plots/dupePlotID.png')
    return 

def plotDupes(dataframe, input1, input2):
    """
    Creates a horizontal bar chart to visualize the effect of deduplication on the total number of data packets in the dataset.

    Args:
        dataframe (pd.DataFrame): DataFrame before deduplication.
        input1 (str): Column name representing sensor IDs.
        input2 (str): Column name representing sensor IDs.

    Returns:
        None
    """
    
    preDedupe = len(dataframe)
    dfDrop = dataframe.drop_duplicates(subset = [input1, input2], inplace = False, ignore_index = True)
    postDedupe = len(dfDrop)
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
    return

def IAThist(df):
    """
    Plots a histogram of the 'Inter Arrival Time' (IAT) column from a DataFrame.

    This function generates and saves a histogram plot of the IAT values 
    present in the provided DataFrame. The histogram is normalized and the 
    x-axis is labeled with 'Inter Arrival Time [in seconds]'. The plot is 
    saved in the '../plots/' directory with a filename based on the DataFrame's 
    name, appended with 'IAThistPlot.png'.

    Args:
        df (pd.DataFrame): DataFrame containing the 'IAT' column for plotting.

    Returns:
        None
    """

    bins = np.linspace(min(df['IAT'].dropna())-0.05*min(df['IAT'].dropna()), max(df['IAT'].dropna())+0.05*max(df['IAT'].dropna()), 10)
    df['IAT'].dropna().plot.hist(bins = bins, edgecolor = 'k', alpha = 0.5, density = True) 
    plt.xticks(rotation = 90)
    plt.xlabel('Inter Arrival Time [in seconds]')
    plt.ylabel('Normalized Frequency Of Occurence')
    plt.savefig('../plots/'+ df.name + 'IAThistPlot.png', bbox_inches = 'tight', transparent = True)
    plt.close()
    return

def outScatterPlot(df):
    """
    Plots a scatter plot of 'Inter Arrival Time' (IAT) values against truncated sensor IDs.

    This function generates a scatter plot with 'idTrunc' on the x-axis and 'IAT' on the y-axis.
    The plot is intended to visualize the distribution and potential outliers of IAT values 
    across different sensors, identified by their truncated IDs.

    Args:
        df (pd.DataFrame): DataFrame containing 'idTrunc' and 'IAT' columns for plotting.

    Returns:
        matplotlib.axes.Axes: Axes object containing the scatter plot.
    """

    plot = df.plot.scatter(x = 'idTrunc', y = 'IAT', rot = 90)
    return plot
    
    
def boxPlot(df, fileName, input1):
    """
    Generates a boxplot of Inter-Arrival Times (IATs) grouped by truncated sensor IDs.

    The boxplot is saved in the '../plots/' directory with a filename based on the DataFrame's name, appended with 'BoxPlot.png'.
    If the dataset is from the Air Quality Monitoring (AQM) project, the sensor IDs are truncated to the last 4 characters for better visibility in the plot.

    Parameters:
        df (pd.DataFrame): DataFrame containing the IAT column and sensor IDs.
        fileName (str): Name of the dataset (used to determine if it's from the AQM project).
        input1 (str): Column name representing the sensor IDs.

    Returns:
        None
    """
    dataName = fileName
#checking to see if dataset is AQM for plot labels (sensor id)
    if "AQM" in dataName:
        df['idTrunc'] = df[input1].str[-4:]
    else:
        df['idTrunc'] = df[input1]
    figure(figsize = (15, 6))
    a = sns.boxplot(x = 'idTrunc', y = df['IAT'], data = df.sort_values(by='IAT', ascending=False, na_position='first'), color = 'seagreen')
    plt.xlabel('Truncated Sensor ID')
    plt.xticks(rotation = 90)
    plt.savefig('../plots/'+ df.name + 'BoxPlot.png', bbox_inches = 'tight', transparent = True)
    plt.close()
    return

def normalFitPlot(df):
    """
    Generates a normal distribution fit plot of Inter-Arrival Times (IATs) along with a histogram.

    The plot is not saved, but the mean and standard deviation of the fitted normal distribution are returned.

    Parameters:
        df (pd.DataFrame): DataFrame containing the IAT column.

    Returns:
        tuple of 2 floats: Mean and standard deviation of the fitted normal distribution.
    """
    
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
    df[df['IAT'] != 0]
    df['IAT'].plot.kde()
    plt.close()
    return mu, std

def piePlot(df, df1, name):
    """
    Generates a pie chart plot to visualize the number of duplicate packets in a given dataset.

    The plot is saved as a PNG image in the '../plots/' directory with a filename based on the dataset name.

    Parameters:
        df (pd.DataFrame): DataFrame containing all data packets.
        df1 (pd.DataFrame): DataFrame containing data packets after removing duplicates.
        name (str): The name of the dataset.

    Returns:
        None
    """
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
    """
    Creates a gauge plot to visualize a single metric score.

    Args:
        metricScore (float): The metric score to visualize
        name (str): The name of the metric

    Returns:
        None
    """
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
    """
    Calculates and plots the outage time per device based on the IAT values.

    Args:
        df (pd.DataFrame): DataFrame containing IAT values and device IDs.
        meanStat (float): Mean statistic of the IAT values.
        stdStat (float): Standard deviation statistic of the IAT values.

    Returns:
        float: Average outage time across all devices.

    Plots:
        A bar chart representing the total outage time per device, saved as 'sensorOutagePlot.png'.
    """

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

def outliersPlot(dataframe):
    """
    Plots a scatter plot of the IAT values with outliers highlighted.

    Uses modified Z-score method to detect outliers, and plots the data with
    outliers marked as 'x's. The threshold for outliers is also plotted as a
    green dashed line.

    Args:
        dataframe (pd.DataFrame): DataFrame containing IAT values

    Plots:
        Scatter plot of IAT values with outliers highlighted, saved as 'outliersPlot.png'.
    """
    df = dataframe
    data = df['IAT'].dropna()

    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z_scores = (0.6745 * (data - median)) / mad
    threshold_mod_z_score = 3.5

    # Identify outliers based on modified z-score threshold
    outliers = df['IAT'][modified_z_scores > threshold_mod_z_score]

    # Plotting the data with outliers highlighted
    plt.figure(figsize=(8, 6))
    plt.scatter(df.index, df['IAT'], color='b', label='Data', alpha=0.5)  # Use alpha to control transparency
    plt.scatter(outliers.index, outliers, color='r', label='Outliers', marker='x')
    plt.axhline(y=(threshold_mod_z_score * mad)/0.6745 + median, color='g', linestyle='--', label='Threshold')
    plt.xlabel('Index')
    plt.ylabel('IAT Values')
    plt.title('IAT Outliers Visualization')
    plt.grid()
    plt.legend()
    plt.savefig("../plots/outliersPlot.png")
    return