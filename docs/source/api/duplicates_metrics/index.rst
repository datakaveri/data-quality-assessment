duplicates_metrics
==================

.. py:module:: duplicates_metrics


Functions
---------

.. autoapisummary::

   duplicates_metrics.duplicatesMetric


Module Contents
---------------

.. py:function:: duplicatesMetric(df, input1, input2)

   Computes duplicate detection metric based on specified columns.
   Must be called before inter-arrival time creation.

   :param df: DataFrame to check for duplicates
   :param input1: First column name to check for duplicates
   :param input2: Second column name to check for duplicates

   :returns: Duplicates metric score (0-1, where 1 means no duplicates).


