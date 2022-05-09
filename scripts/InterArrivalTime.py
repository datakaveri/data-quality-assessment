#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import json
from pandas.io.json import json_normalize
import os
import math
import warnings
import panel as pn

configFile = "../config/" + sys.argv[1]
with open(configFile, "r") as file:
    dataDict = json.load(file)

dataFile = "../data/" + dataDict["dataFileNameJSON"]
with open(dataFile, "r") as jfile:
    jsonDataDict = json.load(jfile)
#print(jsonDataDict)

#df = pd.read_json(dataFile)
df = pd.json_normalize(jsonDataDict)
pd.set_option('mode.chained_assignment', None)

# print(df.shape)


# In[37]:


#dropping duplicates

def dropDupes(dataframe):
    input1 = dataDict['interArrivalTime']['inputFields'][0]
    input2 = dataDict['interArrivalTime']['inputFields'][1]
    dfLen1 = len(dataframe)
    dfDrop = dataframe.drop_duplicates(subset = [input1, input2], inplace = False, ignore_index = True)
    dfLen2 = len(dfDrop)
    dupeCount = dfLen1 - dfLen2
    print(str(dupeCount) + ' duplicate rows have been removed.') 
    print(str(dfDrop.shape) + ' is the shape of the new dataframe.')
    return dfDrop

df1 = dropDupes(df)


# In[53]:


# sorting by unique id and then by oDT

dfCut = df1[[dataDict['interArrivalTime']['inputFields'][0], dataDict['interArrivalTime']['inputFields'][1]]].copy()
dfCut.sort_values(by = [dataDict['interArrivalTime']['inputFields'][0], dataDict['interArrivalTime']['inputFields'][1]], inplace = True, ascending = [True, True])
#dfCut.sort_values(by = ['id','observationDateTime'], inplace = True, ascending = [True,True])
dfCut['observationDateTime'] =  pd.to_datetime(dfCut['observationDateTime'])
dfCut.groupby(by = dataDict['interArrivalTime']['inputFields'][0])
dfCut['IAT'] = dfCut['observationDateTime'].diff()
dfCut = dfCut.reset_index(drop = True)
dfID = (dfCut[dataDict['interArrivalTime']['inputFields'][0]].unique())


# grouping the df by ID and deleting the first row of each group at the same time

dfGroupID = dfCut.groupby(dataDict['interArrivalTime']['inputFields'][0]).apply(lambda group: group.iloc[1:])
dfGroupID = dfGroupID['IAT'].dt.total_seconds()
dfGrouped = dfCut.apply(lambda group: group.iloc[1:])
dfGrouped = dfGrouped['IAT'].dt.total_seconds()

# calculating the mode of the dataset

mode = dfGrouped.mode()[0]

dfOut = dfGrouped.to_frame()
dfOut = dfOut.loc[(dfOut['IAT'] > 0)].reset_index(drop = True)


# In[54]:


# removing outliers outside the range mode +/- 2*mode
outVal = 2 * mode
def removeOutliers(dataframe):
    lenOut = len(dataframe)
    dataframe.drop(dataframe[dataframe > (mode + outVal)].index, inplace = True)
    dataframe.drop(dataframe[dataframe < (mode - outVal)].index, inplace = True)
    dataframe.reset_index(inplace = True, drop = True)
    dfIn = dfGrouped.to_frame()
    lenIn = len(dfIn)
    outRemoved = lenOut - lenIn
    print(str(outRemoved) + ' outliers were removed.')
    print(str(len(dfIn)) + ' rows is the length of the new dataframe.')
    return dataframe

dfIn = removeOutliers(dfGrouped)
# dfGrouped = dfGrouped.to_frame()
# print(dfIn)
# dfIn.mean()
# warnings.filterwarnings("ignore")


# In[40]:


#removing outliers of value >/< mode +/- 2*mode
outliers = []
totalOutage = 0
i = 1

if dataDict['outageFlag'] == 'True':
    bool = input('Would you like to compute the total outage time of all the sensors? [y/n]')
else:
    bool = 'n'
    
if bool == 'y':
    while i < len(dfOut):
            if (dfOut['IAT'][i]) > outVal:
                outliers.append(dfOut['IAT'][i])
                i+=1
            else:
                i+=1
    totalOutage = sum(outliers)
    print("The total outage time of all the sensors combined is: " + str(totalOutage) + " seconds or " + str(round(totalOutage/60,2)) + " minutes")
else:
    print("Outliers of a value greater and less than 2*mode will be removed for a legible output plot.")


# In[48]:


warnings.filterwarnings("ignore")
dfIn.plot.hist(edgecolor = 'k', alpha = 0.5)
vals = plt.gca().get_yticks()
plt.gca().set_yticklabels([round(x/(len(dfIn)),3) for x in vals])
plt.xlabel("Inter-Arrival Time (in seconds)")
plt.ylabel("Frequency as a Ratio of Total Occurences")
plt.savefig('../outputReports/' + dataDict['dataFileNameJSON'] + '_InterArrivalTimePlot.pdf', bbox_inches='tight')  


# In[45]:


#IAT metrics

dfMetric = dfOut
alpha1 = dataDict['interArrivalTime']['alpha'][0]
alpha2 = dataDict['interArrivalTime']['alpha'][1]
alpha3 = dataDict['interArrivalTime']['alpha'][2]
alpha = [alpha1, alpha2, alpha3]
# dfMetric.to_csv('dfmetric.csv')

#calculating range to check metric
i = 0
floor = []
ceil = []
while i < len(alpha):
    floor.append(mode - (alpha[i]*mode))
    ceil.append(mode + (alpha[i]*mode))
    i+=1

# calculating number of values in df outside the desired range
i = 0
j = 1
metricAlpha = [0, 0, 0]

while i < len(floor):
    j = 1
    while j < len(dfMetric):
        if (dfMetric['IAT'][j] < floor[i]) or (dfMetric['IAT'][j] > ceil[i]):
            metricAlpha[i] += 1
            j+=1
        else:
            j+=1
    i+=1 
# print(metricAlpha)

# calculating percentage of values within the desired range
metricPercent = [0,0,0]
i = 0

while i < len(metricAlpha):
    metricPercent[i] = round(((len(dfMetric) - metricAlpha[i])/len(dfMetric))*100,4)
    i+=1
    
# calculating N0 metric
N0 = [0,0,0]
i=0

while i < len(metricPercent):
    N0[i] = round(metricPercent[i]/100, 3)
    i+=1


#print(N0)


# In[46]:


# print statements
warnings.filterwarnings("ignore")
meanOut = round(dfOut.mean().values[0],3)
stdOut = round(dfOut.std().values[0], 3)
#mean, std before outlier removal
print('########################################################################\n')
print(str(meanOut) + " seconds is the average interarrival time of the dataset.")
print(str(stdOut) + " seconds is the standard deviation of the interarrival time of the dataset.")

meanIn = round(dfIn.mean(), 3)
stdIn = round(dfIn.std(), 3)
#mean, std after outlier removal
print(str(meanIn) + " seconds is the average interarrival time after removal of outliers.")
print(str(stdIn) + " seconds is the standard deviation of the interarrival time after removal of outliers.")

#mode
print(str(mode) + " seconds is the mode of the interarrival time of the dataset.")
print("\n########################################################################\n")
print(str(round(metricPercent[0],3)) + "% of data packets lie within the range of (mode +/- alpha*mode) when alpha is: " + str(alpha[0]))
print(str(round(metricPercent[1],3)) + "% of data packets lie within the range of (mode +/- alpha*mode) when alpha is: " + str(alpha[1]))
print(str(round(metricPercent[2],3)) + "% of data packets lie within the range of (mode +/- alpha*mode) when alpha is: " + str(alpha[2]))
print("\n########################################################################")


# In[47]:


#Appending to existing json output file

outputParamIAT = {
    "fileName": dataDict["dataFileNameJSON"],
    "InterArrivalTime":{
        "value": [N0[0], N0[1], N0[2]],
        "value_alpha": dataDict["interArrivalTime"]["alpha"],
        "mean": meanOut,
        "std": stdOut,
        "mode": mode,
        "type": "number",    
        "metricLabel": "InterArrival Time Mode Spread",
        "metricMessage": "There are " + str(metricPercent[0]) + "% data packets that lie inside the range of (mode +/- alpha*mode) when alpha is: " + str(alpha[0]) + ", " + str(metricPercent[1]) + "% when alpha is: " + str(alpha[1]) + " , and " + str(metricPercent[2]) + "% when alpha is: " + str(alpha[2]),
        "description": "The metric is rated on a scale between 0 & 1; Computes the ratio of packets inside the range to the total number of packets."
    }
}
        
myJSON = json.dumps(outputParamIAT, indent = 4)
filename = os.path.splitext(dataDict["fileName"])[0] + "_Report.json"
jsonpath = os.path.join("../outputReports/", filename)

with open(jsonpath, "a+") as jsonfile:
    jsonfile.write(myJSON)
    print("Output file successfully created.")
print("Plot saved as .pdf to outputReports folder")

