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

# mode deviation of dispersion
def compute_mode_deviation(dataframe):
    mode_value = sps.mode(dataframe)[0][0]
    deviations = abs(dataframe - mode_value)
    mode_deviation = deviations.sum() / len(dataframe)
    return mode_deviation

def newRegularityMetric(dataframe):
    dataframe['id'] = dataframe['id'].str[-4:]

    grouped = dataframe.groupby('id')
    # print(grouped)
    result = grouped['IAT'].apply(compute_mode_deviation).reset_index()
    result.columns = ['id', 'mode_deviation']

    # Normalize the mode deviation values between 0 and 1
    scaler = MinMaxScaler()
    result['normalized_mode_deviation'] = scaler.fit_transform(result[['mode_deviation']])
    metricScore = 1 - np.mean(result['normalized_mode_deviation'])
    # print(result)
    print(metricScore)
    return(newRegularityMetric)

newRegularityMetric(dataframe)