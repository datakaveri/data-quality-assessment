#!/usr/bin/env python
# coding: utf-8

# In[40]:


#!/usr/bin/env python
# coding: utf-8

# In[1597]:


#importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re
import sys
import json
from scipy.stats import norm
import statistics
import os
import math
import warnings
import matplotlib.ticker as mtick

warnings.filterwarnings("ignore")

if len(sys.argv) < 2:
    print('###########################################################################')
    print("Not enough arguments")
    print("Usage: python3 InterArrivalTime.py <ConfigFilePath>")
    print('###########################################################################')
    sys.exit()

#Get the data file
#configFile = "../config/PuneAQMConfig.json"
configFile ="../config/" +  sys.argv[1]
#dataFile = sys.argv[2]

#reading from json config file
with open(configFile, "r") as file:
    data_dict = json.load(file)

dataFile = "../data/"+data_dict['fileName']
#parsing Dataset (use nrows attribute to take first n rows)
df = pd.read_csv(dataFile, parse_dates = ['observationDateTime'])

#initiating variables
i=0
j=1
iat = [0]

#parsing Dataset (use nrows attribute to take first n rows)
#df = pd.read_csv(data_dict["fileName"], parse_dates = ['observationDateTime'])


# In[1598]:


#printing details of the dataset for the user input
#print('A sample of the dataset is shown below: ')
#print(df.head())
#print('The column headers in this dataset are: ')
#print(df.columns)
#print('The shape of this dataset is: ' + str(df.shape))
#df['observationDateTime']


# In[1599]:


#dropping duplicate timestamps
df1 = df.drop_duplicates(subset = [data_dict['interArrivalTime']['inputFields'][0], data_dict['interArrivalTime']['inputFields'][1]], inplace = False, ignore_index=True)


# In[1600]:


#creating an array of sorted interarrival times
dfSorted = df1.sort_values(data_dict['interArrivalTime']['inputFields'][1])
iatDict = {k: g['observationDateTime'].tolist() for k,g in dfSorted.groupby(data_dict['interArrivalTime']['inputFields'][0])}
dfIAT = pd.DataFrame.from_dict(iatDict, orient = 'index').transpose()
IATdiff = dfIAT.diff()

#print(IATdiff)
#finding the average IAT for each sensor
avgArr = []
stdArr = []
i = 0
while i < IATdiff.shape[1]:
    avgArr.append(np.mean(IATdiff.iloc[:,i]))
    stdArr.append(np.std(IATdiff.iloc[:,i]))
    i+=1
avgArr = pd.DataFrame(avgArr)
stdArr = pd.DataFrame(stdArr)

avgArr.columns = ['Average IAT for each sensor']
stdArr.columns = ['Std of IAT for each sensor']
overallAvg = np.mean(avgArr)
overallAvg = overallAvg.dt.total_seconds()
overallStd = np.mean(stdArr)
overallStd = overallStd.dt.total_seconds()


# In[1601]:


#creating a plottable array
columnTitles = IATdiff.columns
i = 0
j = 0
plotArr = []
plotRows = []

#creating an array with no. of occurences of each timedelta and sorting
while i < len(columnTitles):
    plotArr.append(IATdiff[columnTitles[i]].value_counts().sort_index())
    i+=1

#converting a list to a dataframe
plotArrDf = pd.DataFrame(plotArr)
rowTitles = plotArrDf.columns


while j < len(rowTitles):
    plotRows.append(plotArrDf[rowTitles[j]])
    j+=1

#functions to round up and down to nearest 10
#def next_ten(x):
#    return int(ceil(x/10.0))*10

#def prev_ten(x):
#    return int(floor(x/10.0))*10

plotArrDf = plotArrDf.sum()
plotArrDf = plotArrDf.sort_index()
plotArrDf.index = plotArrDf.index.total_seconds()
plotArrDf = pd.DataFrame.from_dict(plotArrDf)
plotArrDf.reset_index(level=0, inplace=True)
plotArrDf.columns = ["TimeDelta", "No. Of Occurences"]

#ceil_val = next_ten(plotArrDf['TimeDelta'].max())+10
#floor_val = prev_ten(plotArrDf['TimeDelta'].min())
#print(plotArrDf)


# In[1602]:


#Computing mean,std,mode with outliers

mode_index = plotArrDf["No. Of Occurences"].idxmax()
mode = plotArrDf["TimeDelta"][mode_index]
#print(mode)

# In[1603]:


#Outlier detection
#detecting and summing all timedelta values greater than 2*mode to 
#compute total outage time

twiceMode = 2*mode
outliers = []
i = 0

if data_dict["outageFlag"] == "True":
	bool = input("Would you like to compute the total outage time of all the sensors? [y/n] ")
else:
	bool = 'n'

if bool == 'y':
    while i < len(plotArrDf):
        if (plotArrDf['TimeDelta'][i]) > twiceMode:
            outliers.append(plotArrDf['TimeDelta'][i])
            i+=1
        else:
            i+=1
    totalOutage = (sum(outliers)/(3600))
    print("The total outage time of all the sensors combined is: " + str(round(totalOutage, 2)) + " hours, or " + str(round(totalOutage*60,2)) + " minutes.")
else:
    print("Outliers of a value greater than 2*mode will be removed for a legible output plot.")


# In[1604]:


#Outlier Removal
i = 0

plotArrDfIn = plotArrDf[plotArrDf.TimeDelta < (twiceMode)]

avgArrIn = []
stdArrIn = []
i = 0
#computing the average after removing the outliers
while i < len(plotArrDfIn):
    avgArrIn.append((plotArrDfIn['TimeDelta'][i]*plotArrDfIn['No. Of Occurences'][i]))
    #stdArr.append(np.std(IATdiff.iloc[:,i]))
    i+=1

avgArrIn = sum(avgArrIn)/(plotArrDfIn['No. Of Occurences'].sum())

#computing the standard deviation after removing the outliers
i = 0
while i < len(plotArrDfIn):
    
    stdArrIn.append((plotArrDfIn['TimeDelta'][i] - avgArrIn)*(plotArrDfIn['TimeDelta'][i] - avgArrIn)*plotArrDfIn['No. Of Occurences'][i])
    i+=1
stdArrIn = math.sqrt(sum(stdArrIn)/(plotArrDfIn['No. Of Occurences'].sum()))


# In[1605]:

#Plotting the Array

plotArrDfIndex = plotArrDfIn.set_index('TimeDelta')
#print(plotArrDfIn)

ylabels = []
i = 0
while i < len(plotArrDfIn):
    ylabels = ((plotArrDfIn["No. Of Occurences"]/(plotArrDfIn["No. Of Occurences"].sum()),3))
    i+=1

ax = plt.plot(plotArrDfIn['TimeDelta'],plotArrDfIn['No. Of Occurences'])
vals = plt.gca().get_yticks()
plt.gca().set_yticklabels([round(x/(plotArrDfIn["No. Of Occurences"].sum()),3) for x in vals])
plt.xlabel("Inter-Arrival Time (in seconds)")
plt.ylabel("Ratio of No. of Occurences to Total Data Packets")
plt.savefig('../outputReports/InterArrivalTimeFrequency.pdf', bbox_inches='tight')  


################################################################################################################

#InterArrival Time metrics
alpha1 = data_dict["interArrivalTime"]["alpha"][0]
alpha2 = data_dict["interArrivalTime"]["alpha"][1]
alpha3 = data_dict["interArrivalTime"]["alpha"][2]

metricOut1 = []
metricOut2 = []
metricOut3 = []

#1. No. of data packets within mode +/- alpha*mode
metricDf = plotArrDf

#alpha1
i = 0
while i < len(metricDf):
    if (metricDf['TimeDelta'][i] < (mode - (alpha1*mode))) or (metricDf['TimeDelta'][i] > (mode + (alpha1*mode))):
        metricOut1.append(i)
        i+=1
    else:
        i+=1

#alpha2
i = 0
while i < len(metricDf):
    if (metricDf['TimeDelta'][i] < (mode - (alpha2*mode))) or (metricDf['TimeDelta'][i] > (mode + (alpha2*mode))):
        metricOut2.append(i)
        i+=1
    else:
        i+=1
#alpha3
i = 0
while i < len(metricDf):
    if (metricDf['TimeDelta'][i] < (mode - (alpha3*mode))) or (metricDf['TimeDelta'][i] > (mode + (alpha3*mode))):
        metricOut3.append(i)
        i+=1
    else:
        i+=1

compute1 = []
compute2 = []
compute3 = []

i = 0
while i < len(metricOut1):
    compute1.append(metricDf["No. Of Occurences"][metricOut1[i]])
    i+=1

i=0
while i < len(metricOut2):
    compute2.append(metricDf["No. Of Occurences"][metricOut2[i]])
    i+=1

i=0
while i < len(metricOut3):
    compute3.append(metricDf["No. Of Occurences"][metricOut3[i]])
    i+=1

N0metric1 = 1- (np.sum(compute1)/(metricDf["No. Of Occurences"].sum()))
N0metric2 = 1- (np.sum(compute2)/(metricDf["No. Of Occurences"].sum()))
N0metric3 = 1- (np.sum(compute3)/(metricDf["No. Of Occurences"].sum()))
#print(N0metric)
packetNo1 = (metricDf["No. Of Occurences"].sum()) - np.sum(compute1)
packetNo2 = (metricDf["No. Of Occurences"].sum()) - np.sum(compute2)
packetNo3 = (metricDf["No. Of Occurences"].sum()) - np.sum(compute3)

N0Percent1 = round(N0metric1*100,2)
N0Percent2 = round(N0metric2*100,2)
N0Percent3 = round(N0metric3*100,2)


# In[1607]:

#print statements

print(str(len(df)-len(df1)) + " duplicate rows have been dropped.")
print("The shape of the dataframe is now: " + str(df1.shape))
print("The average inter-arrival time for all the sensors including outliers is: " + str(round(overallAvg.values[0],3)) + " seconds.")
print("The standard deviation of the inter-arrival time for all the sensors including outliers is: " + str(round(overallStd.values[0],3)) + " seconds.")
print("\n")
print("The average inter-arrival time for all the sensors excluding outliers is: " + str(round(avgArrIn,2)) + " seconds.")
print("The standard deviation of all the inter-arrival times excluding outliers is: " + str(round(stdArrIn,2)) + " seconds.")
print("The mode of the inter-arrival times for the sensors is: " + str(round(mode,3)) + " seconds.")

print("########################################################################\n")
print("There are " + str(N0Percent1) + "% data packets that lie inside the range of (mode +/- alpha*mode) when alpha is: " + str(alpha1) + ", " + str(N0Percent2) + "% when alpha is: " + str(alpha2) + " , and " + str(N0Percent3) + "% when alpha is: " + str(alpha3) + ".\n")
print("##########################################################################")


#Appending to existing json output file

outputParamIAT = {
    "fileName": data_dict["fileName"],
    "InterArrivalTime":{
        "value": [(round(N0metric1,4)), (round(N0metric2, 4)), (round(N0metric3, 4))],
        "value_alpha": data_dict["interArrivalTime"]["alpha"],
        "mean": round(overallAvg.values[0],3),
        "std": round(overallStd.values[0],3),
        "mode": mode,
        "type": "number",    
        "metricLabel": "InterArrival Time Mode Spread",
        "metricMessage": "There are " + str(N0Percent1) + "% data packets that lie inside the range of (mode +/- alpha*mode) when alpha is: " + str(alpha1) + ", " + str(N0Percent2) + "% when alpha is: " + str(alpha2) + " , and " + str(N0Percent3) + "% when alpha is: " + str(alpha3),
        "description": "The metric is rated on a scale between 0 & 1; Computes the ratio of packets inside the range to the total number of packets."
    }
}
        
myJSON = json.dumps(outputParamIAT, indent = 4)
filename = os.path.splitext(data_dict["fileName"])[0] + "_Report.json"
jsonpath = os.path.join("../outputReports/", filename)

with open(jsonpath, "a+") as jsonfile:
    jsonfile.write(myJSON)
    print("Output file successfully created.")
print("Plot saved as .pdf to outputReports folder")

