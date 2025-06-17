test_data_preprocessing
=======================

.. py:module:: test_data_preprocessing


Classes
-------

.. autoapisummary::

   test_data_preprocessing.TestPreProcess


Functions
---------

.. autoapisummary::

   test_data_preprocessing.setup_imports


Module Contents
---------------

.. py:function:: setup_imports()

   Setup imports for the test file to work from any location


.. py:class:: TestPreProcess

   Test suite for the preProcess function.


   .. py:method:: test_basic_preprocessing()

      Test basic preprocessing functionality with valid data.



   .. py:method:: test_sorting_functionality()

      Test that data is properly sorted by uniqueID and observationDateTime.



   .. py:method:: test_multiple_unique_ids()

      Test preprocessing with multiple unique IDs.



   .. py:method:: test_negative_iat_filtering()

      Test that negative IAT values are filtered out.



   .. py:method:: test_empty_dataframe()

      Test preprocessing with empty DataFrame.



   .. py:method:: test_single_row_dataframe()

      Test preprocessing with single row DataFrame.



   .. py:method:: test_invalid_datetime_format()

      Test preprocessing with invalid datetime format.



   .. py:method:: test_missing_columns()

      Test preprocessing when required columns are missing.



   .. py:method:: test_nan_datetime_values()

      Test preprocessing with NaN datetime values.



   .. py:method:: test_different_column_names()

      Test preprocessing with different column names for input parameters.



   .. py:method:: test_zero_iat_values()

      Test preprocessing with zero IAT values (same timestamps).



   .. py:method:: test_large_time_differences()

      Test preprocessing with large time differences.



   .. py:method:: test_mixed_data_types_in_unique_id()

      Test preprocessing with mixed data types in uniqueID column.



