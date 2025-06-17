PreProcessing
=============

.. py:module:: PreProcessing


Attributes
----------

.. autoapisummary::

   PreProcessing.custom_style


Functions
---------

.. autoapisummary::

   PreProcessing.readFile
   PreProcessing.timeRange
   PreProcessing.dropDupes
   PreProcessing.outRemove
   PreProcessing.dataStats
   PreProcessing.radarChart
   PreProcessing.bars
   PreProcessing.plotDupesID
   PreProcessing.plotDupes
   PreProcessing.IAThist
   PreProcessing.outScatterPlot
   PreProcessing.boxPlot
   PreProcessing.normalFitPlot
   PreProcessing.piePlot
   PreProcessing.gaugePlot
   PreProcessing.outagePlot
   PreProcessing.outliersPlot


Module Contents
---------------

.. py:data:: custom_style

.. py:function:: readFile(configFile)

   Reads configuration file and data file.

   Reads configuration file and extracts configuration parameters.
   Reads data file and loads data into a pandas DataFrame.

   :param configFile: Path to configuration file
   :type configFile: str

   :returns: Configuration parameters
             dfRaw (pd.DataFrame): Raw data
             input1 (str): First attribute for deduplication
             input2 (str): Second attribute for deduplication
             datasetName (str): Name of the dataset
             fileName (str): Name of the data file
             URL (str): URL link to the dataset
             alpha (list): List of alpha values for IAT regularity metric
             schema (dict or None): Schema for data validation (if schema validation is enabled)
   :rtype: configDict (dict)


.. py:function:: timeRange(dataframe)

   Finds the time range of the dataset.

   :param dataframe: DataFrame with 'observationDateTime' column
   :type dataframe: pd.DataFrame

   :returns: Start time of the dataset
             endTime (pd.Timestamp): End time of the dataset
             startMonth (str): Start month (e.g. Jan)
             endMonth (str): End month (e.g. Dec)
             startYear (int): Start year (e.g. 2020)
             endYear (int): End year (e.g. 2021)
   :rtype: startTime (pd.Timestamp)


.. py:function:: dropDupes(dataframe, input1, input2)

   Drops duplicate rows from a DataFrame based on two columns.

   This function takes a DataFrame and two column names as input. It then
   identifies and removes duplicate rows from the DataFrame based on the two
   columns. The function returns the updated DataFrame and the number of
   duplicate rows that were removed.

   :param dataframe: DataFrame to drop duplicates from
   :type dataframe: pd.DataFrame
   :param input1: First column name to check for duplicates
   :type input1: str
   :param input2: Second column name to check for duplicates
   :type input2: str

   :returns: DataFrame with duplicates removed
             dupeCount (int): Number of duplicate rows that were removed
   :rtype: dfDrop (pd.DataFrame)


.. py:function:: outRemove(df, dataFile, input1)

   Identifies and removes outliers from a DataFrame based on IQR threshold.

   :param df: DataFrame with IAT column
   :type df: pd.DataFrame
   :param dataFile: Name of the data file
   :type dataFile: str
   :param input1: First attribute for deduplication
   :type input1: str

   :returns: DataFrame with outliers removed
             lower (float): Lower bound of the IQR range
             upper (float): Upper bound of the IQR range
   :rtype: dfInliers (pd.DataFrame)


.. py:function:: dataStats(df)

   Compute summary statistics for IAT column in DataFrame.

   :param df: DataFrame with IAT column

   :returns: mean, median, mode, standard deviation, variance, skewness, kurtosis
   :rtype: tuple of 7 integers


.. py:function:: radarChart(regularityScore, outliersScore, dupeScore, formatScore, completeScore, addnlScore)

   Creates a radar chart to visualize the metric scores.

   :param regularityScore: Regularity of InterArrival Time metric score
   :type regularityScore: float
   :param outliersScore: Outliers of Inter-Arrival Time metric score
   :type outliersScore: float
   :param dupeScore: Absence of Duplicates metric score
   :type dupeScore: float
   :param formatScore: Attribute Format Adherence metric score
   :type formatScore: float
   :param completeScore: Mandatory Attribute Adherence metric score
   :type completeScore: float
   :param addnlScore: Unknown Attribute Absence metric score
   :type addnlScore: float

   :returns: None


.. py:function:: bars(score, name)

   Creates a horizontal stacked bar chart to visualize a single metric score.

   :param score: The metric score to visualize
   :type score: float
   :param name: The name of the metric
   :type name: str

   :returns: None


.. py:function:: plotDupesID(df, df1, input1)

   Creates a bar chart to visualize the effect of deduplication on data packets per sensor.

   :param df: Original DataFrame before deduplication.
   :type df: pd.DataFrame
   :param df1: DataFrame after deduplication.
   :type df1: pd.DataFrame
   :param input1: Column name representing sensor IDs.
   :type input1: str

   :returns: None


.. py:function:: plotDupes(dataframe, input1, input2)

   Creates a horizontal bar chart to visualize the effect of deduplication on the total number of data packets in the dataset.

   :param dataframe: DataFrame before deduplication.
   :type dataframe: pd.DataFrame
   :param input1: Column name representing sensor IDs.
   :type input1: str
   :param input2: Column name representing sensor IDs.
   :type input2: str

   :returns: None


.. py:function:: IAThist(df)

   Plots a histogram of the 'Inter Arrival Time' (IAT) column from a DataFrame.

   This function generates and saves a histogram plot of the IAT values
   present in the provided DataFrame. The histogram is normalized and the
   x-axis is labeled with 'Inter Arrival Time [in seconds]'. The plot is
   saved in the '../plots/' directory with a filename based on the DataFrame's
   name, appended with 'IAThistPlot.png'.

   :param df: DataFrame containing the 'IAT' column for plotting.
   :type df: pd.DataFrame

   :returns: None


.. py:function:: outScatterPlot(df)

   Plots a scatter plot of 'Inter Arrival Time' (IAT) values against truncated sensor IDs.

   This function generates a scatter plot with 'idTrunc' on the x-axis and 'IAT' on the y-axis.
   The plot is intended to visualize the distribution and potential outliers of IAT values
   across different sensors, identified by their truncated IDs.

   :param df: DataFrame containing 'idTrunc' and 'IAT' columns for plotting.
   :type df: pd.DataFrame

   :returns: Axes object containing the scatter plot.
   :rtype: matplotlib.axes.Axes


.. py:function:: boxPlot(df, fileName, input1)

   Generates a boxplot of Inter-Arrival Times (IATs) grouped by truncated sensor IDs.

   The boxplot is saved in the '../plots/' directory with a filename based on the DataFrame's name, appended with 'BoxPlot.png'.
   If the dataset is from the Air Quality Monitoring (AQM) project, the sensor IDs are truncated to the last 4 characters for better visibility in the plot.

   :param df: DataFrame containing the IAT column and sensor IDs.
   :type df: pd.DataFrame
   :param fileName: Name of the dataset (used to determine if it's from the AQM project).
   :type fileName: str
   :param input1: Column name representing the sensor IDs.
   :type input1: str

   :returns: None


.. py:function:: normalFitPlot(df)

   Generates a normal distribution fit plot of Inter-Arrival Times (IATs) along with a histogram.

   The plot is not saved, but the mean and standard deviation of the fitted normal distribution are returned.

   :param df: DataFrame containing the IAT column.
   :type df: pd.DataFrame

   :returns: Mean and standard deviation of the fitted normal distribution.
   :rtype: tuple of 2 floats


.. py:function:: piePlot(df, df1, name)

   Generates a pie chart plot to visualize the number of duplicate packets in a given dataset.

   The plot is saved as a PNG image in the '../plots/' directory with a filename based on the dataset name.

   :param df: DataFrame containing all data packets.
   :type df: pd.DataFrame
   :param df1: DataFrame containing data packets after removing duplicates.
   :type df1: pd.DataFrame
   :param name: The name of the dataset.
   :type name: str

   :returns: None


.. py:function:: gaugePlot(metricScore, name)

   Creates a gauge plot to visualize a single metric score.

   :param metricScore: The metric score to visualize
   :type metricScore: float
   :param name: The name of the metric
   :type name: str

   :returns: None


.. py:function:: outagePlot(df, meanStat, stdStat)

   Calculates and plots the outage time per device based on the IAT values.

   :param df: DataFrame containing IAT values and device IDs.
   :type df: pd.DataFrame
   :param meanStat: Mean statistic of the IAT values.
   :type meanStat: float
   :param stdStat: Standard deviation statistic of the IAT values.
   :type stdStat: float

   :returns: Average outage time across all devices.
   :rtype: float

   Plots:
       A bar chart representing the total outage time per device, saved as 'sensorOutagePlot.png'.


.. py:function:: outliersPlot(dataframe)

   Plots a scatter plot of the IAT values with outliers highlighted.

   Uses modified Z-score method to detect outliers, and plots the data with
   outliers marked as 'x's. The threshold for outliers is also plotted as a
   green dashed line.

   :param dataframe: DataFrame containing IAT values
   :type dataframe: pd.DataFrame

   Plots:
       Scatter plot of IAT values with outliers highlighted, saved as 'outliersPlot.png'.


