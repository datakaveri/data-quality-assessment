import ijson
import jsonschema
import json
import sys
import requests
import re
import logging
import os

#Expecting 'data' to be a list
def validate_data_with_schema(dataF, schema):
    num_samples = 0
    err_count = 0
    additional_prop_err_count = 0
    req_prop_err_count = 0
    err_data_arr = []

    with open(dataF, "r") as f:
       for record in ijson.items(f, "item"):
           num_samples = num_samples+1
           data_packet = record
           try:
              jsonschema.validate(data_packet, schema )

           except jsonschema.exceptions.ValidationError as errV:
               logging.debug ("Validation Error Occured")
               #v = jsonschema.Draft7Validator(schema, types=(), format_checker=None)
               v = jsonschema.Draft7Validator(schema)
               errors = sorted(v.iter_errors(data_packet), key=lambda e: e.path)
               if len(errors) > 0:
                   err_count = err_count + 1
               #err_data_arr.append(data_packet)
               #To track if 'Additional Properties' error occured
               flag_0 = 0
               #To track if 'Required Properties' error occured
               flag_1 = 0
               for error in errors:
                   logging.debug (error.message)
                   z = re.match("(Additional properties)", error.message)
                   if z:
                      #logging.debug(z.groups())
                      flag_0 = 1

                   z =  error.message.split(' ')
                   if z[-1] == 'property' and z[-2] == 'required' :
                      flag_1 = flag_1+1

               additional_prop_err_count = additional_prop_err_count + flag_0
               req_prop_err_count = req_prop_err_count + flag_1
           except jsonschema.exceptions.SchemaError as errS:
              logging.debug ("Schema Error Occured")
              logging.debug (errS.message)

    return num_samples, err_count, err_data_arr, additional_prop_err_count, req_prop_err_count

def validate_requiredFields(dataF, setReqd):
    num_samples = 0
    num_missing_prop = 0
    with open(dataF, "r") as f:
       #Read each record instead of reading all at a time
       for record in ijson.items(f, "item"):
           num_samples = num_samples+1
           #setRecd = record.keys()
           setRecd = []
           #Null value detection. Null values are considered as not received attribute
           for attr in record.keys():
               if record[attr] is None:
                   logging.debug("Received a Null Value for attribute: " + attr)
               else:
                   setRecd.append(attr)
           diffSet = set(setReqd) - set(setRecd)
           logging.debug("Difference from Required Fields for this packet: "+str(diffSet))
           print("Difference from Required Fields for this packet: "+str(diffSet))
           num_missing_prop = num_missing_prop + len(diffSet)
       return num_samples, num_missing_prop

#Main program
if len(sys.argv) < 2:
    print('###########################################################################')
    print("Not enough arguments")
    print("Usage: python3 validate_format <ConfiFilePath>")
    print('###########################################################################')
    sys.exit()

#logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

#Get Config File
configFile = sys.argv[1]
with open(configFile, "r") as cf:
    config = json.load(cf)

#Get the data file
#dataFile = sys.argv[1]
dataFile = "../data/"+config['dataFileNameJSON']

#Get the schema file
#schemaFile = sys.argv[2]
schemaFile = "../schemas/"+config['schemaFileName']

#Load the data file
#with open(dataFile, "r") as f:
#    data = json.load(f)

#Load the schema file
with open(schemaFile, "r") as f1:
    schema = json.load(f1)


#####################################################################
# Format Validity for attributes for which data formats are provided
# in the Schema
# For this metric we will ignore
#   1. errors that occur because "required" fields are not there
#   2. errors that occur because additional fields are present
#
#####################################################################

# Remove Required properties and Additional Properties from Schema
del schema['additionalProperties']
del schema['required']

num_samples, err_count, err_data_arr, add_err_count, req_err_cnt = validate_data_with_schema(dataFile, schema)

#logging.debug(err_data_arr)
logging.info('###########################################################################')
logging.info("Total Samples: " + str(num_samples))
logging.info("Total Format Errors: " + str(err_count))
format_adherence_metric = 1 - err_count/num_samples
logging.info("Format Adherence Metric: " + str(format_adherence_metric))
logging.info('###########################################################################')


#######################################################################
# Unknown data attribute metric: Fraction of data points for
# which contain only known fields:(1 - fraction(datapoints that contain
# unknown attribtues))
# in the Schema
# For this metric we will ignore
#   1. errors that occur because of format issues
#   2. errors that occur because of required fields not present
#
#######################################################################
with open(schemaFile, "r") as f1:
    schema = json.load(f1)

#Remove Required properties and Additional Properties from Schema
del schema['required']
#NOTE: Should we explicitly make additional properties as false
schema['additionalProperties'] = False

num_samples, err_count, err_data_arr, add_err_count, req_err_cnt = validate_data_with_schema(dataFile, schema)

logging.debug(err_data_arr)
logging.info("Total samples: " + str(num_samples))
logging.info("Total Additional Fields Error Count: " + str(add_err_count))
unknown_fields_absent_metric = 1 - add_err_count/num_samples
logging.info("Unknown_Attributes_Absent_Metric: " + str(unknown_fields_absent_metric))

#######################################################################
# One by one check the required properties are present in packets or
# not
#######################################################################
with open(schemaFile, "r") as f1:
    schema = json.load(f1)

del schema['additionalProperties']
req = schema['required']
logging.debug(len(req))
missing_attr = {}
completeness_metric = 0
num_samples, total_missing_count = validate_requiredFields(dataFile, req)

logging.info("Total missing count: " + str(total_missing_count))


completeness_metric = 1 - total_missing_count/(num_samples*len(req))

logging.info('###########################################################################')
logging.info('##### Total Missing Fields Count for Required fields #######')
logging.info("Total samples: " + str(num_samples))
logging.info("Attribute_Completeness_Metric: "+str(completeness_metric))
logging.info('###########################################################################')



logging.info('################## Final Metrics ##########################################')
logging.info("Format Adherence Metric: " + str(format_adherence_metric))
logging.info("Additional Fields Absent Metric: " + str(unknown_fields_absent_metric))
logging.info("Attribute Completeness Metric: "+str(completeness_metric))
logging.info('###########################################################################')



#Outputting the result to a json report

outputParamFV = {
    "FormatAdherence":{
        "value": str(format_adherence_metric),
        "type": "number",
        "metricLabel": "Format Adherence Metric",
        "metricMessage": "For this dataset, " + str(format_adherence_metric) + " is the format adherence",
        "description": "The metric is rated on a scale between 0 & 1; Computes the ratio of data packets with attributes that adhere to the format defined in the data schema."
        },
    "AdditionalFieldsAbsent":{
        "value": str(unknown_fields_absent_metric),
        "type": "number",
        "metricLabel": "Unknown Fields Absent Metric",
        "metricMessage": "For this dataset, " + str(unknown_fields_absent_metric) + " is the additional fields absent metric.",
        "description": "The metric is rated on a scale between 0 & 1; computed as (1 - r) where r is the ratio of packets with unknown fields to the total number of packets."
        },
    "AttributeCompleteness":{
        "value": str(completeness_metric),
        "type": "number",
        "metricLabel": "Completeness Metric",
        "metricMessage": "For this dataset, " + str(completeness_metric) + " is the completeness metric.",
        "description": "The metric is rated on a scale between 0 & 1; It is computed as follows: For each mandatory attribute, i, compute r(i) as the ratio of packets in which attribute i is missing. Then output 1 - average(r(i)) where the average is taken over all mandatory attributes."
        }
}
myJSON = json.dumps(outputParamFV, indent = 4)
filename = os.path.splitext(config["fileName"])[0] + "_Report.json"
jsonpath = os.path.join("../outputReports/",filename)

with open(jsonpath, "w") as jsonfile:
    jsonfile.write(myJSON)
    print("Output file successfully created.")
