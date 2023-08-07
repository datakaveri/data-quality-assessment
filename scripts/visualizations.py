import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as sps
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import PreProcessing as pp

# reading the file
dataframe, input1, input2, datasetName, fileName, URL, alpha, schema = pp.readFile('../config/puneAQMConfig.json')

# dropping duplicates
dataframe, dupeCount = pp.dropDupes(dataframe, input1, input2)

# getting the inter-arrival times
dataframe = pp.preProcess(dataframe, input1, input2)

# print(dataframe.head())

def iatViolinAllSensors(dataframe):
    # Trim the data by capping the IAT values at the threshold of 920
    df_trimmed = dataframe.copy()  # Create a copy of the original DataFrame
    df_trimmed.loc[df_trimmed['IAT'] > 920, 'IAT'] = 920

    # Set the style for the plot
    sns.set(style='whitegrid')  # Choose a desired style, e.g., 'whitegrid'

    # Create a violin plot with the trimmed data and additional styling options
    sns.violinplot(data=df_trimmed, x='IAT', inner='quartile', palette='pastel', linewidth=1.5, saturation=0.8, bw = 0.5)

    # Set plot labels and title
    plt.xlabel('IAT Values [in seconds]', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.title('Distribution of IAT Values Across All Sensors', fontsize=12)

    # Set the font size of tick labels
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    # Adjust the plot dimensions and spacing
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    # Display the plot
    plt.show()
    return

def iatViolinMultiSensors(dataframe):
    
    # Filter the data for three different sensors
    dataframe['idTrunc'] = dataframe['id'].str[-4:]

    sensor1_data = dataframe[dataframe['idTrunc'] == 'bd9f']['IAT'].dropna()
    sensor2_data = dataframe[dataframe['idTrunc'] == 'bf3d']['IAT'].dropna()
    sensor3_data = dataframe[dataframe['idTrunc'] == 'ea3c']['IAT'].dropna()
    sensor4_data = dataframe[dataframe['idTrunc'] == '5f4a']['IAT'].dropna()
    sensor5_data = dataframe[dataframe['idTrunc'] == 'f153']['IAT'].dropna()


    # Combine the three datasets into a single DataFrame
    combined_data = pd.concat([sensor1_data, sensor2_data, sensor3_data], axis=1)
    test_data = pd.concat([sensor1_data, sensor2_data, sensor3_data, sensor4_data ,sensor5_data], axis=1)
    combined_data.columns = ['Sensor 1', 'Sensor 2', 'Sensor 3']
    test_data.columns = ['Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4', 'Sensor 5']


    # Create the violin plot with the combined data
    sns.set(style='whitegrid')
    sns.violinplot(data=test_data, inner='quartile', palette='pastel', linewidth=1.5, saturation=0.8, bw = 0.5)

    # Set plot labels and title
    plt.xlabel('Sensors', fontsize=10)
    plt.ylabel('IAT Values [in seconds]', fontsize=10)
    plt.title('Distribution of IAT Values for Five Sensors', fontsize=12)

    # Set the font size of tick labels
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    # Adjust the plot dimensions and spacing
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    # Display the plot
    plt.show()
    return

# iatViolinMultiSensors(dataframe)


# computing a new metric for regularity
# mode deviation of dispersion
def computeModeDeviation(dataframe):
    modeValue = sps.mode(dataframe)[0][0]
    deviations = abs(dataframe - modeValue)
    modeDeviation = deviations.sum() / len(dataframe)
    return modeDeviation

def newRegularityMetric(dataframe):
    dataframe['id'] = dataframe['id'].str[-4:]

    grouped = dataframe.groupby('id')
    # print(grouped)
    result = grouped['IAT'].apply(computeModeDeviation).reset_index()
    result.columns = ['id', 'mode_deviation']

    # Normalize the mode deviation values between 0 and 1
    scaler = MinMaxScaler()
    result['normalized_mode_deviation'] = scaler.fit_transform(result[['mode_deviation']])
    metricScore = 1 - np.mean(result['normalized_mode_deviation'])
    # print(result)
    print(metricScore)
    return(newRegularityMetric)

def modeRegularityMetric(dataframe):
    data = dataframe['IAT'].dropna()
    modeValue = sps.mode(data)[0][0]
    # grouped = dataframe.groupby('id')
    lower = modeValue - (modeValue/2)
    upper = modeValue + (modeValue/2)
    print(lower,upper)
    outliers = [x for x in data if x < lower or x > upper] 
    outlierNumber = len(outliers)
    totalDataPackets = len(data)
    print(outlierNumber, totalDataPackets)
    iatMetricOutlierScore = 1 - (outlierNumber/totalDataPackets)
    print(iatMetricOutlierScore)
    return

def RAERegularityMetric(dataframe):

    dataframe = dataframe.drop_duplicates(['id', 'observationDateTime'])
    dataframe = dataframe[(dataframe['IAT'] < 901) & (dataframe['IAT'] > 899)]
    data = dataframe['IAT'].dropna()
    modeValue = sps.mode(data)[0][0]
    RAE = np.absolute((data - modeValue)/modeValue)
    lower = modeValue - (modeValue/2)
    upper = modeValue + (modeValue/2)
    print(RAE.max(), RAE.min())
    goodCount = 0
    badCount = 0
    count = 0
    for iat in data:
        RAE = np.absolute((iat - modeValue)/modeValue)
        if RAE <= 0.5:
            goodCount += 1 - (2*RAE)
            count += 1
        else:
            badCount += 2*RAE
            print(iat)
        totalCount = count + badCount
    
    print(totalCount, count, goodCount, badCount)
    RAERegularityMetric = np.round(goodCount/totalCount, 3) 
    print(RAERegularityMetric)
    return

RAERegularityMetric(dataframe)

def iatOutliersMetricIQR(dataframe):
    print('combination of IQR and modified z-score method')
    df = dataframe
    data = df['IAT'].dropna()
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)

    # adaptive threshold using regular z-score
    median = np.median(data)
    mad = np.median(np.abs(data - median))

    modified_z_scores = (0.6745 * (data - median)) / mad # applicable to normal symmetric distribution

    # threshold_z_score = np.mean(modified_z_scores)
    threshold_z_score = np.percentile(np.abs(modified_z_scores), 95)
    # k = 1.5
    IQR = Q3 - Q1
    cutOff = IQR*threshold_z_score
    # cutOff = IQR*k
    lower, upper = Q1 - cutOff, Q3 + cutOff
    print(lower,upper)
    outliers = [x for x in df['IAT'] if x < lower or x > upper] 
    outlierNumber = len(outliers)
    totalDataPackets = len(df)
    print(outlierNumber, totalDataPackets)
    iatMetricOutlierScore = 1 - (outlierNumber/totalDataPackets)
    print(iatMetricOutlierScore)

    # visualisations
    # normally distributed having removed outliers
    data = pd.DataFrame(data)
    modeValue = sps.mode(data)[0][0]
    # dataNorm = data[~data.IAT.isin(outliers)]
    dataNorm = data - modeValue
    dataNorm = dataNorm[(dataNorm > 0).all(1)]
    print(dataNorm)
    print(modeValue)
    # plt.hist(dataNorm['IAT'], alpha = 0.5, edgecolor = 'k')
    sns.histplot(data=dataNorm['IAT'], element = "step", kde=True, log_scale = True)
    # plt.plot(dataNorm['IAT'], linestyle = 'dotted')
    plt.xlabel('IAT')
    plt.ylabel('Count')
    plt.title('Exponentially Distributed IAT Values')
    plt.legend()
    plt.show()

    #outlier viz with thresholds (UF, LF)
    # visualising the outliers 
    plt.figure(figsize=(8, 6))
    # plt.scatter(data.index, data, color='b', label='Data')
    plt.scatter(data['IAT'].index, data['IAT'], color = 'b', label='IAT',linewidths=0.1)
    # plt.scatter(outliers_z.index, outliers_z, color='r', label='Outliers')
    plt.plot(data['IAT'].index, [lower] * len(data['IAT']), color='r', linestyle='--', label='Lower Fence', linewidth=1)
    plt.plot(data['IAT'].index, [upper] * len(data['IAT']), color='r', linestyle='--', label='Upper Fence', linewidth=1)
    plt.xlabel('Count')
    plt.ylabel('IAT')
    plt.title('Outliers Detection Visualization')
    # plt.legend()
    # plt.show()

    return iatMetricOutlierScore


# iglewicz and hoaglin
def iatOutliersMetricZscore(dataframe):
    print('modified z-score method proposed by iglewicz and hoaglin')
    data = dataframe['IAT'].dropna()
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z_scores = (0.6745 * (data - median)) / mad  # Calculate modified Z-scores
    threshold = np.percentile(np.abs(modified_z_scores), 95)
    print(threshold)
    outliers_z = [x for x in modified_z_scores if x > threshold] # can also be changed to 3.5 as per recommendation of iglewicz and hoaglin
    # outliers_z = [x for x in modified_z_scores if x > 3.5]
    # showing normal fit plot
    # print(modified_z_scores)
    data = pd.DataFrame(data)
    data['modzscores'] = modified_z_scores
    dataNorm = data[~data.modzscores.isin(outliers_z)]

    plt.hist(dataNorm['IAT'], alpha = 0.5, edgecolor = 'k')

    # visualising the outliers 
    plt.figure(figsize=(8, 6))
    # plt.scatter(data.index, data, color='b', label='Data')
    plt.scatter(modified_z_scores.index, modified_z_scores, color = 'b', label='ModZScores',linewidths=0.1)
    # plt.scatter(outliers_z.index, outliers_z, color='r', label='Outliers')
    plt.plot(modified_z_scores.index, [threshold] * len(modified_z_scores), color='r', linestyle='--', label='Threshold', linewidth=1)
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Outliers Visualization')
    plt.legend()
    # plt.show()

    outlierNumber = len(outliers_z)
    totalDataPackets = len(data)
    print(outlierNumber, totalDataPackets)
    iatMetricOutlierScore = 1 - (outlierNumber/totalDataPackets)
    print(iatMetricOutlierScore)
    return iatMetricOutlierScore


# duplicate detection metrics
def duplicatesMetric(df):
    totalDataPackets = len(df)
    dfDupes = df.duplicated(keep='first')
    dupeCount = dfDupes.value_counts()[True]
    duplicatesMetricScore = 1 - dupeCount/totalDataPackets
    # print(str(dupeMetricPercent) + '% of the data packets are non duplicates, as defined by the parameters ' + str(input1) ' &' + str(input2))
    print(duplicatesMetricScore)
    return round(duplicatesMetricScore,3)

newRegularityMetric(dataframe)
# iatOutliersMetricIQR(dataframe)
# modeRegularityMetric(dataframe)
# iatOutliersMetricZscore(dataframe)

# duplicatesMetric(dataframe1)
# pp.normalFitPlot(dataframe)
