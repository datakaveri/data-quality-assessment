import ijson
import logging

def validate_requiredFields(dataF, setReqd):
    """
    Validates that required fields are present in JSON data.
    Treats null values as missing attributes.
    
    Args:
        dataF: Path to JSON file containing data
        setReqd: Set of required field names
        
    Returns:
        Tuple containing:
<<<<<<< HEAD
        - num_samples: Total number of samples processed.
        - num_missing_prop: Total number of missing properties across all samples.
=======
        - num_samples: Total number of samples processed
        - num_missing_prop: Total number of missing properties across all samples
>>>>>>> 9c5f2989031ba54019bec835b7ecb3f5768f2dcf
    """
    num_samples = 0
    num_missing_prop = 0
    with open(dataF, "r") as f:
        # Read each record instead of reading all at a time
        for record in ijson.items(f, "item"):
            num_samples = num_samples + 1
            setRecd = []
            # Null value detection. Null values are considered as not received attribute
            for attr in record.keys():
                if record[attr] is None:
                    logging.debug("Received a Null Value for attribute: " + attr)
                else:
                    setRecd.append(attr)
            diffSet = set(setReqd) - set(setRecd)
            logging.debug("Difference from Required Fields for this packet: " + str(diffSet))
            num_missing_prop = num_missing_prop + len(diffSet)
    return num_samples, num_missing_prop