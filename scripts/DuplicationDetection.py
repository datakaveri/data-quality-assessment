#!/usr/bin/env python
# coding: utf-8

# In[58]:


#importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json
import os
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print('###########################################################################')
    print("Not enough arguments")
    print("Usage: python3 DuplicationDetection.py <ConfiFilePath>")
    print('###########################################################################')
    sys.exit()

#Get the data file
#configFile = "config.json"
configFile = "../config/" +  sys.argv[1]
#dataFile = sys.argv[2]


#reading from json config file
with open(configFile, "r") as file:
    data_dict = json.load(file)

dataFile = "../data/" + data_dict['fileName']
#parsing Dataset (use nrows attribute to take first n rows)
df = pd.read_csv(dataFile, parse_dates = ['observationDateTime'])


#reading from json config file
#with open("config.json") as file:
#    data_dict = json.load(file)

#parsing Dataset (use nrows attribute to take first n rows)
#df = pd.read_csv(data_dict['fileName'], parse_dates = ['observationDateTime'])


# In[59]:


#printing details of the dataset for the user input
#print('A sample of the dataset is shown below: ')
#print(df.head())
#print('The column headers in this dataset are: ')
#print(df.columns)
#print('The shape of this dataset is: (Rows, Columns) ')
#print(df.shape)


# In[60]:


#Count Number of duplicates

dupeCount = len(df)-len(df.drop_duplicates(subset = [data_dict['duplicateDetection']["inputFields"][0], data_dict['duplicateDetection']['inputFields'][1]]))


# In[42]:


#drop duplicate timestamps
#bool = input("Would you like to drop the duplicates from the dataset? [y/n] ")
#if bool == 'y':
    #df1 = df.drop_duplicates(subset = [data_dict['duplicateDetection']['inputFields'][0], data_dict['duplicateDetection']['inputFields'][1]], inplace = False, ignore_index=True)
    #print('The length of the dataset after removing the duplicate rows from the columns ' + data_dict['duplicateDetection']["inputFields"][0] + ' & ' + data_dict['duplicateDetection']["inputFields"][1] + ' is: ' + str(df1.shape))
#else:
    #df1 = df
    #print('The length of the dataset without removing the duplicate rows from the columns ' + data_dict['duplicateDetection']["inputFields"][0] + ' & ' + data_dict['duplicateDetection']["inputFields"][1] + ' is: ' + str(df1.shape))


# In[61]:


#Calculating Duplication metric
dupeMetric = 1- (dupeCount/len(df))
dupePercent = round(dupeMetric*100,2)
print("#######################################################################\n")
print('The number of duplicate rows in the dataset is: ' + str(dupeCount))
print('\n')
print("The metric score for duplicates is: " + str(dupeMetric) + " or " + str(dupePercent) + "%")
print('\n')
print("#######################################################################\n")




#Outputting the result to a json report
listObj = []

outputParamDD = {
    "fileName": data_dict["fileName"],
    "duplicateDetection":{
    "value": (round(dupeMetric,3)),
    "type": "number",    
    "metricLabel": "Duplicate Count Metric",
    "metricMessage": "For this dataset, " + str(dupePercent) + "% of the data packets are NOT duplicated.",
    "description": "The metric is rated on a scale between 0 & 1; Computes (1- r) where r is the ratio of duplicate packets to the total number of packets." 
    }
}

#print(os.path.splitext(data_dict["fileName"])[0])

myJSON = json.dumps(outputParamDD, indent = 4)
filename = os.path.splitext(data_dict["fileName"])[0] + "_Report.json"
jsonpath = os.path.join("../outputReports/",filename)

with open(jsonpath, "a+") as jsonfile:
    jsonfile.write(myJSON)
    print("Output file successfully created.")
