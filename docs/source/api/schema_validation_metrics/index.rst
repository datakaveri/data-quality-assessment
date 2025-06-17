schema_validation_metrics
=========================

.. py:module:: schema_validation_metrics


Functions
---------

.. autoapisummary::

   schema_validation_metrics.validate_data_with_schema


Module Contents
---------------

.. py:function:: validate_data_with_schema(dataF, schema)

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



