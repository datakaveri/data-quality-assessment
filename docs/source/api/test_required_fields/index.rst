test_required_fields
====================

.. py:module:: test_required_fields


Classes
-------

.. autoapisummary::

   test_required_fields.TestRequiredFieldsValidation


Functions
---------

.. autoapisummary::

   test_required_fields.setup_imports


Module Contents
---------------

.. py:function:: setup_imports()

   Setup imports for the test file to work from any location


.. py:class:: TestRequiredFieldsValidation

   .. py:method:: required_fields_set()

      Sample set of required fields



   .. py:method:: complete_data_file()

      Create a temporary file with complete data (all required fields present)



   .. py:method:: missing_fields_data_file()

      Create a temporary file with missing required fields



   .. py:method:: null_values_data_file()

      Create a temporary file with null values (treated as missing)



   .. py:method:: mixed_data_file()

      Create a temporary file with mixed scenarios



   .. py:method:: teardown_method()

      Clean up temporary files



   .. py:method:: test_all_fields_present(required_fields_set, complete_data_file)

      Test when all required fields are present in all records



   .. py:method:: test_missing_fields(required_fields_set, missing_fields_data_file)

      Test when some required fields are missing



   .. py:method:: test_null_values_treated_as_missing(required_fields_set, null_values_data_file)

      Test that null values are treated as missing attributes



   .. py:method:: test_mixed_scenarios(required_fields_set, mixed_data_file)

      Test mixed scenarios with complete, missing, and null values



   .. py:method:: test_empty_required_fields_set(complete_data_file)

      Test with empty set of required fields



   .. py:method:: test_empty_data_file(required_fields_set)

      Test with empty JSON array



   .. py:method:: test_single_record_all_missing(required_fields_set)

      Test with single record missing all required fields



   .. py:method:: test_single_record_all_null(required_fields_set)

      Test with single record where all required fields are null



   .. py:method:: test_file_not_found(required_fields_set)

      Test with non-existent file



   .. py:method:: test_large_required_fields_set(complete_data_file)

      Test with large set of required fields



   .. py:method:: test_extra_fields_ignored(required_fields_set)

      Test that extra fields (not in required set) are ignored



   .. py:method:: test_partial_overlap_required_fields()

      Test with required fields that partially overlap with data fields



