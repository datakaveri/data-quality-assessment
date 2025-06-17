test_iat_outliers
=================

.. py:module:: test_iat_outliers


Classes
-------

.. autoapisummary::

   test_iat_outliers.TestIATOutliersMetric


Functions
---------

.. autoapisummary::

   test_iat_outliers.setup_imports


Module Contents
---------------

.. py:function:: setup_imports()

   Setup imports for the test file to work from any location


.. py:class:: TestIATOutliersMetric

   .. py:method:: test_no_outliers_perfect_score()

      Test that data with no outliers returns score of 1.0



   .. py:method:: test_all_outliers_zero_score()

      Test that data with all outliers returns score close to 0



   .. py:method:: test_mixed_data_with_outliers()

      Test with mixed data containing some outliers



   .. py:method:: test_single_value()

      Test with single IAT value



   .. py:method:: test_two_identical_values()

      Test with two identical IAT values



   .. py:method:: test_two_different_values()

      Test with two different IAT values



   .. py:method:: test_with_nan_values()

      Test that NaN values are properly handled



   .. py:method:: test_empty_dataframe_after_dropna()

      Test behavior with dataframe that becomes empty after dropping NaN



   .. py:method:: test_normal_distribution()

      Test with normally distributed data



   .. py:method:: test_return_type_and_precision()

      Test that return value is properly rounded to 3 decimal places



   .. py:method:: test_zero_mad_edge_case()

      Test edge case where MAD might be zero



   .. py:method:: test_large_dataset()

      Test with a larger dataset



   .. py:method:: test_negative_values()

      Test with negative IAT values



   .. py:method:: test_very_small_values()

      Test with very small IAT values



   .. py:method:: sample_dataframe()

      Fixture providing a sample dataframe for testing



   .. py:method:: test_with_additional_columns(sample_dataframe)

      Test that function works with dataframes containing additional columns



