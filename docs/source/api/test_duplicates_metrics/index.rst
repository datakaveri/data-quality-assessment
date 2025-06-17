test_duplicates_metrics
=======================

.. py:module:: test_duplicates_metrics


Classes
-------

.. autoapisummary::

   test_duplicates_metrics.TestDuplicatesMetric


Functions
---------

.. autoapisummary::

   test_duplicates_metrics.setup_imports


Module Contents
---------------

.. py:function:: setup_imports()

   Setup imports for the test file to work from any location


.. py:class:: TestDuplicatesMetric

   Test suite for the duplicatesMetric function.


   .. py:method:: test_no_duplicates()

      Test metric calculation when there are no duplicates.



   .. py:method:: test_all_duplicates()

      Test metric calculation when all rows are duplicates.



   .. py:method:: test_partial_duplicates()

      Test metric calculation with some duplicates.



   .. py:method:: test_single_row_dataframe()

      Test metric calculation with single row DataFrame.



   .. py:method:: test_empty_dataframe()

      Test metric calculation with empty DataFrame.



   .. py:method:: test_two_rows_no_duplicates()

      Test metric calculation with two unique rows.



   .. py:method:: test_two_rows_all_duplicates()

      Test metric calculation with two identical rows.



   .. py:method:: test_nan_values_in_columns()

      Test metric calculation with NaN values in specified columns.



   .. py:method:: test_different_column_names()

      Test metric calculation with different column names.



   .. py:method:: test_numeric_columns()

      Test metric calculation with numeric columns.



   .. py:method:: test_mixed_data_types()

      Test metric calculation with mixed data types.



   .. py:method:: test_large_dataset_performance()

      Test metric calculation with larger dataset.



   .. py:method:: test_rounding_precision()

      Test that the result is properly rounded to 3 decimal places.



   .. py:method:: test_missing_columns_error()

      Test error handling when specified columns don't exist.



   .. py:method:: test_duplicate_detection_ignores_other_columns()

      Test that duplicate detection only considers specified columns.



