#!/usr/bin/env python
# coding: utf-8

# In[8]:


#Configuration file (.ini)

from configparser import ConfigParser

config_object = ConfigParser()
userfile_name = input("Which data file would you like to open? ")
usercolumn_name = input("Which column would you like to scrape for duplicates? ")

config_object["DUPLICATEDETECTION"] = {
    "filename": userfile_name,
    "column_name": usercolumn_name
}

with open('config.ini', 'w') as conf:
    config_object.write(conf)

