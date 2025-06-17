test_iat_regularity
===================

.. py:module:: test_iat_regularity


Classes
-------

.. autoapisummary::

   test_iat_regularity.TestComputeModeDeviation
   test_iat_regularity.TestIATRegularityMetric


Functions
---------

.. autoapisummary::

   test_iat_regularity.setup_imports


Module Contents
---------------

.. py:function:: setup_imports()

   Setup imports for the test file to work from any location


.. py:class:: TestComputeModeDeviation

   .. py:method:: test_identical_values()

      Test mode deviation with identical values



   .. py:method:: test_simple_deviation()

      Test mode deviation with simple known values



   .. py:method:: test_single_value()

      Test mode deviation with single value



   .. py:method:: test_float_values()

      Test mode deviation with float values



.. py:class:: TestIATRegularityMetric

   .. py:method:: test_perfect_regularity()

      Test with perfectly regular IAT values (all same)



   .. py:method:: test_high_regularity()

      Test with high regularity (small deviations from mode)



   .. py:method:: test_low_regularity()

      Test with low regularity (large deviations from mode)



   .. py:method:: test_single_value()

      Test with single IAT value



   .. py:method:: test_two_identical_values()

      Test with two identical values



   .. py:method:: test_mixed_regularity()

      Test with mixed regularity



   .. py:method:: test_return_type_and_precision()

      Test that return value is properly rounded to 3 decimal places



   .. py:method:: test_with_additional_columns()

      Test that function works with dataframes containing additional columns



