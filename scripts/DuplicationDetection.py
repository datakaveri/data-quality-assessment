#!/usr/bin/env python
# coding: utf-8

# In[75]:


#importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json

#reading from json config file
with open("Config.json") as file:
    data_dict = json.load(file)
#print(data_dict['duplicateDetection'][0])

#reading from config.ini file
#config_object = ConfigParser()
#config_object.read("config.ini")
#dd = config_object["DUPLICATEDETECTION"]
#filename = dd["filename"]
#column_dupe = dd["column_name"]


#parsing Dataset (use nrows attribute to take first n rows)
df = pd.read_csv(data_dict['duplicateDetection'][0]['fileName'], parse_dates = ['observationDateTime'])


# In[76]:


#printing details of the dataset for the user input
print('A sample of the dataset is shown below: ')
print(df.head())
print('The column headers in this dataset are: ')
print(df.columns)
print('The shape of this dataset is: (Rows, Columns) ')
print(df.shape)
#print(df["location.coordinates"][0][0])


# In[77]:


#Count Number of duplicates

dupeCount = len(df)-len(df.drop_duplicates(subset = [data_dict['duplicateDetection'][1]["columnName1"], data_dict['duplicateDetection'][2]['columnName2']]))
print('The number of duplicate rows in the dataset is: ' + str(dupeCount))


# In[78]:


#drop duplicate timestamps
bool = input("Would you like to drop the duplicates from the dataset? [y/n] ")
if bool == 'y':
    df1 = df.drop_duplicates(subset = [data_dict['duplicateDetection'][1]["columnName1"], data_dict['duplicateDetection'][2]["columnName2"]], inplace = False,ignore_index=True)
    print('The length of the dataset after removing the duplicate rows from the columns ' + data_dict['duplicateDetection'][1]["columnName1"] + ' & ' + data_dict['duplicateDetection'][2]["columnName2"] + ' is: ' + str(df1.shape))
else:
    df1 = df
    print('The length of the dataset without removing the duplicate rows from the columns ' + data_dict['duplicateDetection'][1]["columnName1"] + ' & ' + data_dict['duplicateDetection'][2]["columnName2"] + ' is: ' + str(df1.shape))


# In[95]:


#Calculating Duplication metric
dupeMetric = 1- (dupeCount/len(df))
dupePercent = round(dupeMetric*100,2)
print("The metric score for duplicates is: " + str(dupeMetric) + " or " + str(dupePercent) + "%")


# In[102]:


#Outputting the result to a json report

outputParam = {
    "duplicateDetection":[
    {"fileName": data_dict["duplicateDetection"][0]["fileName"]},
    {"Metric Score": str(round(dupeMetric,3))},
    {"Metric Percent": str(dupePercent) + '%'},
    {"Metric Label": "duplicate detection"},
    {"Metric Message": "The metric is rated on a scale between 0 & 1; 1 being the highest possible score."}    
    ]
}

myJSON = json.dumps(outputParam, indent = 4)

with open("Report.json", "w") as jsonfile:
    jsonfile.write(myJSON)
    print("Output file successfully created.")

