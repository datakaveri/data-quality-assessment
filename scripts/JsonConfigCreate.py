#!/usr/bin/env python
# coding: utf-8

# In[45]:


import json
import pandas as pd

#initializing and clearing variables
userfileName = ''
usercolumnName1 = ''
usercolumnName2 = ''

#file format validation
def formatCheck(prompt = 'Which file would you like to open? '):
    userfileName = input(prompt)
    if userfileName[-4:] in ('.csv'): return userfileName
    return formatCheck(prompt = 'The file must be in .csv format. Please retry: ')

#column existence validation
def columnCheck(prompt = 'Which columns would you like to check for duplicates? Please enter the first followed by the second column name: '):
    usercolumnName = input(prompt)
    if usercolumnName in df: return usercolumnName
    return columnCheck(prompt = 'This column does not exist in the dataset. Please retry: ')

#reading the file and checking for exceptions
while True:
    userfileName = formatCheck()
    try:
        df = pd.read_csv(userfileName, parse_dates = ['observationDateTime'])
    except:
        print("This file does not exist. Please enter a valid file name. ")
    else:
        break

usercolumnName1 = columnCheck()
userColumnName2 = columnCheck()

inputParam = {
    "duplicateDetection":[
    {"fileName": userfileName},
    {"columnName1": usercolumnName1},
    {"columnName2": userColumnName2}    
    ]
}

myJSON = json.dumps(inputParam, indent = 4)

with open("Config.json", "w") as jsonfile:
    jsonfile.write(myJSON)
    print("Configuration file successfully created.")


# In[ ]:





# In[ ]:




