#!/usr/bin/env python
# coding: utf-8

# In[28]:


import json
import pandas as pd

#initializing and clearing variables
userfile_name = ''
usercolumn_name = ''

#file format validation
def format_check(prompt = 'Which file would you like to open? '):
    userfile_name = input(prompt)
    if userfile_name[-4:] in ('.csv'): return userfile_name
    return format_check(prompt = 'The file must be in .csv format. Please retry: ')

#column existence validation
def column_check(prompt = 'Which column would you like to search for duplicates? '):
    usercolumn_name = input(prompt)
    if usercolumn_name in df: return usercolumn_name
    return column_check(prompt = 'This column does not exist in the dataset. Please retry: ')

#reading the file and checking for exceptions
while True:
    userfile_name = format_check()
    try:
        df = pd.read_csv(userfile_name, parse_dates = ['observationDateTime'])
    except:
        print("This file does not exist. Please enter a valid file name. ")
    else:
        break

usercolumn_name = column_check()

input_param = {
    "filename": userfile_name,
    "column_name": usercolumn_name
}

myJSON = json.dumps(input_param)

with open("Config.json", "w") as jsonfile:
    jsonfile.write(myJSON)
    print("Configuration file successfully created.")


# In[ ]:





# In[ ]:




