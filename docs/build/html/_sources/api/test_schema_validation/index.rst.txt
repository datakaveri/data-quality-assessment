test_schema_validation
======================

.. py:module:: test_schema_validation


Classes
-------

.. autoapisummary::

   test_schema_validation.TestSchemaValidationMetrics


Functions
---------

.. autoapisummary::

   test_schema_validation.setup_imports


Module Contents
---------------

.. py:function:: setup_imports()

   Setup imports for the test file to work from any location


.. py:class:: TestSchemaValidationMetrics

   .. py:method:: sample_schema()

      Sample JSON schema for testing



   .. py:method:: valid_data_file()

      Create a temporary file with valid JSON data



   .. py:method:: invalid_data_file()

      Create a temporary file with invalid JSON data



   .. py:method:: mixed_errors_file()

      Create a temporary file with mixed validation errors



   .. py:method:: teardown_method()

      Clean up temporary files



   .. py:method:: test_validate_all_valid_data(sample_schema, valid_data_file)

      Test validation with all valid data



   .. py:method:: test_validate_with_errors(sample_schema, invalid_data_file)

      Test validation with various error types



   .. py:method:: test_validate_mixed_errors(sample_schema, mixed_errors_file)

      Test validation with mixed error types



   .. py:method:: test_empty_file(sample_schema)

      Test validation with empty JSON array



   .. py:method:: test_invalid_schema(valid_data_file)

      Test with invalid schema



   .. py:method:: test_file_not_found(sample_schema)

      Test with non-existent file



   .. py:method:: test_required_property_error_detection(sample_schema)

      Test specific detection of required property errors



   .. py:method:: test_additional_property_error_detection(sample_schema)

      Test specific detection of additional property errors



   .. py:method:: test_complex_schema_validation()

      Test with a more complex schema



