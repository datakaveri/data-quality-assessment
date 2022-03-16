#!/usr/bin/env python
# coding: utf-8

# In[18]:


#importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from configparser import ConfigParser

#reading from config.ini file
config_object = ConfigParser()
config_object.read("config.ini")
dd = config_object["DUPLICATEDETECTION"]
filename = dd["filename"]
column_dupe = dd["column_name"]
#initiating variables
i=0
j=1
iat = [0]

#parsing Dataset (use nrows attribute to take first n rows)
df = pd.read_csv(filename, parse_dates = ['observationDateTime'])


# In[19]:


#printing details of the dataset for the user input
print('A sample of the dataset is shown below: ')
print(df.head())
print('The column headers in this dataset are: ')
print(df.columns)
print('The shape of this dataset is: (Rows, Columns) ')
print(df.shape)


# In[16]:


#Count Number of duplicates
#column_dupe =  input('Which column do you want to search for duplicates? ')

dupe_count = len(df)-len(df.drop_duplicates(column_dupe))
print('The number of duplicate rows in the dataset is: ' + str(dupe_count))


# In[17]:


#drop duplicate timestamps
df1 = df.drop_duplicates(subset = column_dupe,inplace = False,ignore_index=True)
print('The length of the dataset after removing the duplicate rows from the column ' + column_dupe + ' is: ' + str(df1.shape))

