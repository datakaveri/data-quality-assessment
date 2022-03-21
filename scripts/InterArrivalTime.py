#!/usr/bin/env python
# coding: utf-8

# In[188]:


#importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re
import sys
import json
from os import path

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


# In[189]:


#printing details of the dataset for the user input
#print('A sample of the dataset is shown below: ')
#print(df.head())
#print('The column headers in this dataset are: ')
#print(df.columns)
#print('The shape of this dataset is: (Rows, Columns) ')
print(df.shape)
df['observationDateTime']


# In[190]:


#dropping duplicate timestamps
df1 = df.drop_duplicates(subset = [data_dict['interArrivalTime']['inputFields'][0], data_dict['interArrivalTime']['inputFields'][1]], inplace = False, ignore_index=True)
print(str(len(df)-len(df1)) + " duplicate rows have been dropped.")
print(df1.shape)
df1['observationDateTime']


# In[191]:


dfSorted = df1.sort_values(data_dict['interArrivalTime']['inputFields'][1])
#print(dfSorted)
iatDict = {k: g['observationDateTime'].tolist() for k,g in dfSorted.groupby("trip_id")}
#print(iatDict)
dfIAT = pd.DataFrame.from_dict(iatDict, orient = 'index').transpose()


# In[346]:


i = 0
j = 0
IATdiff = dfIAT.diff()
avg = np.mean(IATdiff)
std = np.std(IATdiff)
maxi = np.max(IATdiff)

print("The average inter-arrival time for all the sensors is: " + str(np.mean(avg)))
print("The standard deviation of the inter-arrival time for all the sensors combined is: " + str(np.mean(std)))


# In[372]:


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
    
plotRowsDf = pd.DataFrame(plotRows)
plotArrDf = plotArrDf.sum()
#plotArrDf.sort_index
plot = plotArrDf.sort_index().plot.bar()
#plot = plotArrDf.iloc[0,:].sort_index().plot.bar()
#plot.set_xlabel("Inter-Arrival Time")
#plot.set_ylabel("No. of Occurences")


# In[ ]:





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




