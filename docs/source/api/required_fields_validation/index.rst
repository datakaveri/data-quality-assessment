required_fields_validation
==========================

.. py:module:: required_fields_validation


Functions
---------

.. autoapisummary::

   required_fields_validation.validate_requiredFields


Module Contents
---------------

.. py:function:: validate_requiredFields(dataF, setReqd)

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



