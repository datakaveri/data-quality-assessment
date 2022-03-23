#!/usr/bin/env python
# coding: utf-8

# In[926]:


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
from os import path
from math import ceil, floor

#Get the data file
#configFile = sys.argv[1]
#dataFile = sys.argv[2]

#reading from json config file
with open("config.json", "r") as file:
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


# In[927]:


#printing details of the dataset for the user input
#print('A sample of the dataset is shown below: ')
#print(df.head())
print('The column headers in this dataset are: ')
print(df.columns)
print('The shape of this dataset is: ' + str(df.shape))
#df['observationDateTime']


# In[928]:


#dropping duplicate timestamps
df1 = df.drop_duplicates(subset = [data_dict['interArrivalTime']['inputFields'][0], data_dict['interArrivalTime']['inputFields'][1]], inplace = False, ignore_index=True)
print(str(len(df)-len(df1)) + " duplicate rows have been dropped.")
print("The shape of the dataframe is now: " + str(df1.shape))


# In[929]:


dfSorted = df1.sort_values(data_dict['interArrivalTime']['inputFields'][1])
#print(dfSorted)
#dfSorted.to_csv('varibable.csv')
iatDict = {k: g['observationDateTime'].tolist() for k,g in dfSorted.groupby(data_dict['interArrivalTime']['inputFields'][0])}
#print(iatDict)
dfIAT = pd.DataFrame.from_dict(iatDict, orient = 'index').transpose()
#dfIAT.to_csv('sensor_test.csv')
IATdiff = dfIAT.diff()


# In[930]:


#creating a plottable array
columnTitles = IATdiff.columns
i = 0
j = 0
Arr = []
plotArr = []
plotRows = []
while i < len(columnTitles):
    plotArr.append(IATdiff[columnTitles[i]].value_counts().sort_index())
    i+=1
    
plotArrDf = pd.DataFrame(plotArr)
rowTitles = plotArrDf.columns

while j < len(rowTitles):
    plotRows.append(plotArrDf[rowTitles[j]])
    j+=1

#function to round up to nearest 10
def next_ten(x):
    return int(ceil(x/10.0))*10

def prev_ten(x):
    return int(floor(x/10.0))*10
    
plotRowsDf = pd.DataFrame(plotRows)
#print(plotArrDf)
plotArrDf = plotArrDf.sum()
plotArrDf = plotArrDf.sort_index()
plotArrDf.index = plotArrDf.index.total_seconds()
plotArrDf = pd.DataFrame.from_dict(plotArrDf)
plotArrDf.reset_index(level=0, inplace=True)
plotArrDf.columns = ["TimeDelta", "No. Of Occurences"]
#print(len(plotArrDf['TimeDelta']))
#print(len(plotArrDf['No. Of Occurences']))
#ceil_val = next_ten(plotArrDf.max())+10
#floor_val = prev_ten(plotArrDf.min())
ceil_val = next_ten(plotArrDf['TimeDelta'].max())+10
floor_val = prev_ten(plotArrDf['TimeDelta'].min())

#print(plotArrDf)
#plotArrDf.to_csv('varibable.csv')


# In[961]:


#Computing mean,std,mode with outliers
print(plotArrDf["TimeDelta"])
avg = np.mean(plotArrDf["TimeDelta"])/60
std = np.std(plotArrDf["TimeDelta"])/60
maxi = np.max(plotArrDf["TimeDelta"])/60
print("The average inter-arrival time for all the sensors including outliers is: " + str(round(avg,2)) + " minutes.")
print("The standard deviation of the inter-arrival time for all the sensors including outliers is: " + str(round(std,2)) + " minutes.")

mode_index = plotArrDf["No. Of Occurences"].idxmax()
mode = plotArrDf["TimeDelta"][mode_index]/60


# In[932]:


#Outlier detection
#detecting and summing all timedelta values greater than 2*mode to 
#compute total outage time

twiceMode = 2*mode
outliers = []
i = 0

bool = input("Would you like to compute the total outage time of all the sensors? [y/n] ")
if bool == 'y':
    while i < len(plotArrDf):
        if (plotArrDf['TimeDelta'][i])/60 > twiceMode+1:
            outliers.append(plotArrDf['TimeDelta'][i]/60)
            i+=1
        else:
            i+=1
    totalOutage = (sum(outliers)/(60*24))
    print("The total outage time of all the sensors combined is: " + str(round(totalOutage, 2)) + " days")
else:
    print("Outliers of a value greater than 2*mode will be removed for a legible output plot.")


# In[933]:


#Outlier Removal
i = 0
#print(twiceMode)
#print(plotArrDf)
plotArrDfIn = plotArrDf[plotArrDf.TimeDelta < (twiceMode*60)+1]
#print(plotArrDfIn)


# In[934]:


#Plotting the Array

#plotArrDfIndex = plotArrDfIn.set_index('TimeDelta')
plot = plotArrDfIn['No. Of Occurences'].plot.bar()
plt.xlabel("Inter-Arrival Time (in minutes)")
plt.ylabel("No. of Occurences")
xlabels = round(plotArrDfIn.TimeDelta/60,2)
plot.set_xticklabels(xlabels)
plt.xticks(rotation = 90, ha="right")
#plot = plt.bar(plotArrDfIn['TimeDelta']/60,plotArrDfIn['No. Of Occurences'])
plt.show()


#floor_val
#ceil_val
#bins = np.arange(floor_val,ceil_val,15)
#plt.hist(plotArrDf, bins=np.arange(floor_val,ceil_val,15))
#plot = plt.hist(plotArrDf, bins=40, alpha=0.5)
#plt.show()


# In[959]:


#Computing mean,std,mode without outliers

avgIn = np.mean(plotArrDfIn["TimeDelta"])/60
stdIn = np.std(plotArrDfIn["TimeDelta"])/60
maxIn = np.max(plotArrDfIn["TimeDelta"])/60

print("The average inter-arrival time for all the sensors with outliers removed is: " + str(round(avgIn,2)) + " minutes.")
print("The standard deviation of the inter-arrival time for all the sensors combined with outliers removed is: " + str(round(stdIn,2)) + " minutes.")
print("The mode of the inter-arrival time for all the sensors is: " + str(mode) + " minutes")


# In[986]:


#InterArrival Time metrics
i = 0
alpha = 0.1
metricOut = []
#1. No. of data packets within alpha*std +/- mode
metricDf = plotArrDf
#print(std)
#print(mode + alpha*std)
#print(metricDf)
while i < len(metricDf):
    if (metricDf['TimeDelta'][i]/60 < (mode - (alpha*std))) or (metricDf['TimeDelta'][i]/60 > (mode + (alpha*std))):
        metricOut.append(i)
        i+=1
    else:
        i+=1
print("There are " + str(len(metricDf) - len(metricOut)) + " data packets that lie outside the range of (mode +/- alpha*std)")
print("Here, alpha is: " + str(alpha))
N_0 = (len(metricDf)-len(metricOut))/(len(metricDf))
print(N_0)


# In[46]:


#Appending to existing json output file
IATOutput = []

outputParamIAT = {
    "fileName": data_dict["fileName"],
    "InterArrivalTime":{
    "value": 5,
    "type": "number",    
    "metricLabel": "Duplicate Count Metric",
    "metricMessage": "For this dataset, % of the data packets are duplicates.",
    "description": "The metric is rated on a scale between 0 & 1; Computes the ratio of duplicate packets."
    }
}

f = data_dict["fileName"] + "Report.json"
#checking if file exists
if path.isfile(f) is False:
    raise Exception("File not found")
    
myJSON = json.dumps(IATOutput, indent = 4)
#reading JSON file
with open(f,'r+') as fp:
    IATOutput = json.load(fp)
    IATOutput.update(outputParamIAT)
    #fp.seek(0)
    fp.write(myJSON)
    #for i in range(3):
       #IATOutput.append(outputParamIAT[[(x, x**3) for x in range(1,3)]])
    #json.dumps(IATOutput, indent = 4)


# In[ ]:




