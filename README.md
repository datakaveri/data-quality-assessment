# Data Quality Assessment Tool

The scripts included in this repository can be used as a way to audit the quality of a dataset through 4 metrics. These metrics are used to evaluate non-sensor data and validate the dataset against a provided JSON-LD schema document. In order to assess the quality of sensor data, please switch to the main branch. 

A PDF report is generated as the output along with a JSON file. These reports contain the result of the evaluation procedure described below:

For each of the metrics that are used to quantify the quality of a dataset, the tool aims to provide a score between 0 and 1, where 1 is the highest possible score, indicating a 100% score.
Currently, the tool is able to assess four parameters, namely:

- Absence of Duplicate Values
- Adherence to Attribute Format
- Absence of Unknown Attributes
- Adherence to Mandatory Attributes

A note to remember is that each dataset has a linked JSON Schema that defines the attributes that can be present in the dataset, whether there are any required attributes, and what units and datatypes these attributes need to have. Additionally, each dataset must have a config file assoociated with it to generate the output reports. The config files are included in this repository.

### Duplicate Presence

This metric serves to check two columns that are input by the user for any duplicate values in the dataset. A value is considered to be a duplicate if both columns contain the exact same values for any data packet. 

### Attribute Format Adherence

This metric assesses the level of adherence of the data to its expected format as defined in the data schema.
It is quantified by taking the ratio of packets adhering to the expected schemas to the total number of data packets.

### Absence of Unknown Attributes

This metric checks whether there are any additional attributes present in the dataset apart from the list of required attributes.

### Adherence to Mandatory Attributes

This metric checks whether all the required attributes defined in the schema are present in the dataset.


Some additional information about the dataset is also provided in the report. A more detailed description and evaluation criteria for all these metrics is provided in the output PDF report.

## Generating Reports
The first step is to ensure that the IUDX SDK is installed on your computer using the following command:

### Running the tool
Clone the repo from:

``` console
git clone https://github.com/datakaveri/data-quality-assessment.git
```

### Choosing the appropriate branch
In order to evaluate non-sensor generated or non-temporal datasets, switch to the "adex" branch.

### Required libraries and packages
Once in the scripts folder, run the following command to install the package and library dependencies:

```console
pip install -r requirements.txt
```

Present in the *config* folder is a config file in *JSON* format with the name of the dataset prepended to it. This file requires one to input the name of the datafile as well as select the attributes that one would like to check for duplicates. Ensure that the parameters folderName, dataFileNameJSON, schemaFileName are correctly filled according to the dataset to be analyzed. Additional parameters can also be changed, such as inputFields for 'duplicateDetection' as well as the value of 'alpha' - used to identify outliers. Ensure that all these parameters are appropriately set prior to running the main script.

Present in the *data* folder in the repository are some sample datasets. Inside the *schemas* folder are the corresponding schemas for these datasets. In order to generate the report, simply run the following command:

```console
python3 DQReportGenerator.py

```
and enter the name of the selected config file when prompted.

Ensure that the datasets are in *JSON* format and are located in the *data* folder.

The output report file will be generated in a *JSON* format and and a *.pdf* format and will be saved in the outputReports folder. These files will be prepended by the name of the dataset as taken from the config file. The plots and visualizations required for generating the PDF report will be stored in the *plots* folder and will be populated as the script is running. These will be overwritten everytime the script is run and do not need to be stored locally long-term.
