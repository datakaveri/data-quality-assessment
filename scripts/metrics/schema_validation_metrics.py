import ijson
import jsonschema
import fastjsonschema
import logging
import re

def validate_data_with_schema(dataF, schema):
    """
    Validates JSON data against a given schema and tracks various error types.
    
    Args:
<<<<<<< HEAD
        dataF: Path to JSON file containing data.
        schema: JSON schema to validate against.
        
    Returns:
        Tuple containing:
        - num_samples: Total number of samples processed.
        - err_count: Number of samples with validation errors.
        - err_data_arr: Array of data packets with errors.
        - additional_prop_err_count: Count of additional properties errors.
        - req_prop_err_count: Count of required properties errors.
=======
        dataF: Path to JSON file containing data
        schema: JSON schema to validate against
        
    Returns:
        Tuple containing:
        - num_samples: Total number of samples processed
        - err_count: Number of samples with validation errors
        - err_data_arr: Array of data packets with errors
        - additional_prop_err_count: Count of additional properties errors
        - req_prop_err_count: Count of required properties errors
>>>>>>> 9c5f2989031ba54019bec835b7ecb3f5768f2dcf
    """
    num_samples = 0
    err_count = 0
    additional_prop_err_count = 0
    req_prop_err_count = 0
    err_data_arr = []

    # First, try to validate the schema itself
    try:
        # Attempt to compile the schema to catch definition errors early
        fastjsonschema.compile(schema)
    except fastjsonschema.exceptions.JsonSchemaDefinitionException as schema_def_err:
        logging.error(f"Schema definition error: {schema_def_err}")
        # Return zeros for all metrics when schema is invalid
        return 0, 0, [], 0, 0
    except Exception as general_schema_err:
        logging.error(f"General schema error: {general_schema_err}")
        # Return zeros for all metrics when schema is invalid
        return 0, 0, [], 0, 0

    with open(dataF, "r") as f:
        for record in ijson.items(f, "item"):
            num_samples = num_samples + 1
            data_packet = record
            try:
                fastjsonschema.validate(schema, data_packet)

            except fastjsonschema.exceptions.JsonSchemaValueException as errV:
                logging.debug("Validation Error Occured")
                v = jsonschema.Draft7Validator(schema)
                errors = list(v.iter_errors(data_packet))
                if len(errors) > 0:
                    err_count = err_count + 1
                    err_data_arr.append(data_packet)
                flag_0 = 0
                # To track if 'Required Properties' error occured
                flag_1 = 0
                for error in errors:
                    logging.debug(error.message)
                    z = re.match("(Additional properties)", error.message)
                    if z:
                        flag_0 = 1

                    z = error.message.split(' ')
                    if z[-1] == 'property' and z[-2] == 'required':
                        flag_1 = flag_1 + 1

                additional_prop_err_count = additional_prop_err_count + flag_0
                req_prop_err_count = req_prop_err_count + flag_1
                
            except fastjsonschema.exceptions.JsonSchemaDefinitionException as errD:
                logging.error(f"Schema Definition Error: {errD}")
                # If we get a definition error during validation, it means the schema is invalid
                # Return current counts up to this point
                return num_samples, err_count, err_data_arr, additional_prop_err_count, req_prop_err_count
                
            except jsonschema.exceptions.SchemaError as errS:
                logging.debug("Schema Error Occured")
                logging.debug(errS.message)

    return num_samples, err_count, err_data_arr, additional_prop_err_count, req_prop_err_count