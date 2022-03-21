#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json

#reading from json config file
with open("config.json") as file:
    data_dict = json.load(file)

#parsing Dataset (use nrows attribute to take first n rows)
df = pd.read_csv(data_dict['fileName'], parse_dates = ['observationDateTime'])


# In[2]:


#printing details of the dataset for the user input
print('A sample of the dataset is shown below: ')
print(df.head())
print('The column headers in this dataset are: ')
print(df.columns)
print('The shape of this dataset is: (Rows, Columns) ')
print(df.shape)
#print(df["location.coordinates"][0][0])


# In[3]:


#Count Number of duplicates

dupeCount = len(df)-len(df.drop_duplicates(subset = [data_dict['duplicateDetection']["inputFields"][0], data_dict['duplicateDetection']['inputFields'][1]]))
print('The number of duplicate rows in the dataset is: ' + str(dupeCount))


# In[5]:


#drop duplicate timestamps
#bool = input("Would you like to drop the duplicates from the dataset? [y/n] ")
#if bool == 'y':
    #df1 = df.drop_duplicates(subset = [data_dict['duplicateDetection']['inputFields'][0], data_dict['duplicateDetection']['inputFields'][1]], inplace = False, ignore_index=True)
    #print('The length of the dataset after removing the duplicate rows from the columns ' + data_dict['duplicateDetection']["inputFields"][0] + ' & ' + data_dict['duplicateDetection']["inputFields"][1] + ' is: ' + str(df1.shape))
#else:
    #df1 = df
    #print('The length of the dataset without removing the duplicate rows from the columns ' + data_dict['duplicateDetection']["inputFields"][0] + ' & ' + data_dict['duplicateDetection']["inputFields"][1] + ' is: ' + str(df1.shape))


# In[6]:


#Calculating Duplication metric
dupeMetric = 1- (dupeCount/len(df))
dupePercent = round(dupeMetric*100,2)
print("The metric score for duplicates is: " + str(dupeMetric) + " or " + str(dupePercent) + "%")


# In[13]:


#Outputting the result to a json report
listObj = []

outputParamDD = {
    "fileName": data_dict["fileName"],
    "duplicateDetection":{
    "value": (round(dupeMetric,3)),
    "type": "number",    
    "metricLabel": "Duplicate Count Metric",
    "metricMessage": "For this dataset, " + str(dupePercent) + "% of the data packets are duplicates.",
    "description": "The metric is rated on a scale between 0 & 1; Computes the ratio of duplicate packets."
    }
}

print(type(outputParamDD))

myJSON = json.dumps(outputParamDD, indent = 4)

with open(data_dict["fileName"] + "Report.json", "w") as jsonfile:
    jsonfile.write(myJSON)
    print("Output file successfully created.")


# In[ ]:




