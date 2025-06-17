iat_outliers_metrics
====================

.. py:module:: iat_outliers_metrics


Functions
---------

.. autoapisummary::

   iat_outliers_metrics.iatOutliersMetric


Module Contents
---------------

.. py:function:: iatOutliersMetric(dataframe)

   Computes IAT outliers metric using modified Z-score approach.
   Uses median absolute deviation (MAD) for robust outlier detection.

   :param dataframe: DataFrame with 'IAT' column

   :returns: IAT outliers metric score (0-1, where 1 means no outliers)


