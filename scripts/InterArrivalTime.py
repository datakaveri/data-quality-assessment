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
from math import ceil, floor, sqrt

#Get the data file
#configFile = "config.json"
configFile = sys.argv[1]
#dataFile = sys.argv[2]

#reading from json config file
with open(configFile, "r") as file:
    data_dict = json.load(file)

dataFile = "../data/"+data_dict['fileName']
#parsing Dataset (use nrows attribute to take first n rows)
df = pd.read_csv(data_dict["fileName"], parse_dates = ['observationDateTime'])

#initiating variables
i=0
j=1
iat = [0]

#parsing Dataset (use nrows attribute to take first n rows)
df = pd.read_csv(data_dict["fileName"], parse_dates = ['observationDateTime'])


# In[1598]:


#printing details of the dataset for the user input
#print('A sample of the dataset is shown below: ')
#print(df.head())
print('The column headers in this dataset are: ')
print(df.columns)
print('The shape of this dataset is: ' + str(df.shape))
#df['observationDateTime']


# In[1599]:


#dropping duplicate timestamps
df1 = df.drop_duplicates(subset = [data_dict['interArrivalTime']['inputFields'][0], data_dict['interArrivalTime']['inputFields'][1]], inplace = False, ignore_index=True)
print(str(len(df)-len(df1)) + " duplicate rows have been dropped.")
print("The shape of the dataframe is now: " + str(df1.shape))


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
def next_ten(x):
    return int(ceil(x/10.0))*10

def prev_ten(x):
    return int(floor(x/10.0))*10

plotArrDf = plotArrDf.sum()
plotArrDf = plotArrDf.sort_index()
plotArrDf.index = plotArrDf.index.total_seconds()
plotArrDf = pd.DataFrame.from_dict(plotArrDf)
plotArrDf.reset_index(level=0, inplace=True)
plotArrDf.columns = ["TimeDelta", "No. Of Occurences"]

ceil_val = next_ten(plotArrDf['TimeDelta'].max())+10
floor_val = prev_ten(plotArrDf['TimeDelta'].min())
print(plotArrDf)


# In[1602]:


#Computing mean,std,mode with outliers
print("The average inter-arrival time for all the sensors including outliers is: " + str(round(overallAvg.values[0],3)) + " seconds.")
print("The standard deviation of the inter-arrival time for all the sensors including outliers is: " + str(round(overallStd.values[0],3)) + " seconds.")

mode_index = plotArrDf["No. Of Occurences"].idxmax()
mode = plotArrDf["TimeDelta"][mode_index]
print("The mode of the inter-arrival times for the sensors is: " + str(round(mode,3)) + " seconds")


# In[1603]:


#Outlier detection
#detecting and summing all timedelta values greater than 2*mode to 
#compute total outage time

twiceMode = 5*mode
outliers = []
i = 0

bool = input("Would you like to compute the total outage time of all the sensors? [y/n] ")
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
    print("Outliers of a value greater than 5*mode will be removed for a legible output plot.")


# In[1604]:


#Outlier Removal
i = 0
#print(twiceMode)
#print(plotArrDf)
plotArrDfIn = plotArrDf[plotArrDf.TimeDelta < (twiceMode)]
#print(plotArrDfIn)
plotArrDfIn.to_csv('plotArrDfIn.csv')
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
print("The average inter-arrival time for all the sensors excluding outliers is: " + str(round(avgArrIn,2)) + " seconds.")
print("The standard deviation of all the inter-arrival times excluding outliers is: " + str(round(stdArrIn,2)) + " seconds.")


# In[1605]:


#Plotting the Array

out = pd.cut(plotArrDfIn.TimeDelta, len(plotArrDfIn.TimeDelta), right = True, ordered = True)
#print(out)
plotArrDfIndex = plotArrDfIn.set_index('TimeDelta')
#print(plotArrDfIn)

xlabels = out

#print(plotArrDfIn)

ylabels = []
i = 0
while i < len(plotArrDfIn):
    ylabels = ((plotArrDfIn["No. Of Occurences"]/(plotArrDfIn["No. Of Occurences"].sum()),3))
    i+=1

ax = plotArrDfIn['No. Of Occurences'].plot.line()
type(ax)
vals = ax.get_yticks()
ax.set_yticklabels([round(x/(plotArrDfIn["No. Of Occurences"].sum()),3) for x in vals])


#xlabels = round(plotArrDfIn.TimeDelta,3)
#ax.set_xticklabels(xlabels)

plt.xticks(rotation = 45, ha="right")
plt.xlabel("Inter-Arrival Time (in seconds)")
plt.ylabel("No. of Occurences")
plot.figure.savefig('InterArrivalTimeFrequency.pdf', bbox_inches='tight')  


# In[1606]:


#InterArrival Time metrics
i = 0
alpha = data_dict["interArrivalTime"]["alpha"][2]
metricOut = []
#1. No. of data packets within alpha*std +/- mode
metricDf = plotArrDf

while i < len(metricDf):
    if (metricDf['TimeDelta'][i] < (mode - (alpha*mode))) or (metricDf['TimeDelta'][i] > (mode + (alpha*mode))):
        metricOut.append(i)
        i+=1
    else:
        i+=1

compute = []
i = 0
while i < len(metricOut):
    compute.append(metricDf["No. Of Occurences"][metricOut[i]])
    i+=1
N0metric = 1- (np.sum(compute)/(metricDf["No. Of Occurences"].sum()))
print(N0metric)
packetNo = (metricDf["No. Of Occurences"].sum()) - np.sum(compute)
print("There are " + str(packetNo) + " data packets that lie inside the range of (mode +/- alpha*mode)")
print("Here, alpha is: " + str(alpha))


# In[1607]:


#Appending to existing json output file

outputParamIAT = {
    "fileName": data_dict["fileName"],
    "InterArrivalTime":{
        "value": round(N0metric,4),
        "value_alpha": data_dict["interArrivalTime"]["alpha"],
        "mean": round(overallAvg.values[0],3),
        "std": round(overallStd.values[0],3),
        "mode": mode,
        "type": "number",    
        "metricLabel": "InterArrival Time Mode Spread",
        "metricMessage": "For this dataset, " + str(packetNo) + " data packets lie within the range of (mode +/- alpha*mode), where alpha is: " + str(alpha),
        "description": "The metric is rated on a scale between 0 & 1; Computes the ratio of packets inside the range to the total number of packets."
    }
}
        
myJSON = json.dumps(outputParamIAT, indent = 4)

with open(os.path.splitext(data_dict["fileName"])[0] + "_Report.json", "a+") as jsonfile:
    jsonfile.write(myJSON)
    print("Output file successfully created.")

