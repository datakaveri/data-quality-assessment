iat_regularity_metrics
======================

.. py:module:: iat_regularity_metrics


Functions
---------

.. autoapisummary::

   iat_regularity_metrics.computeModeDeviation
   iat_regularity_metrics.iatRegularityMetric


Module Contents
---------------

.. py:function:: computeModeDeviation(dataframe)

       Computes mode deviation for a given dataframe series.

       Args:
   <<<<<<< HEAD
           dataframe: Series or array of values.
   =======
           dataframe: Series or array of values
   >>>>>>> 9c5f2989031ba54019bec835b7ecb3f5768f2dcf

       Returns:
           Mode deviation value



.. py:function:: iatRegularityMetric(dataframe)

   Computes IAT regularity metric using Relative Absolute Error (RAE).

   :param dataframe: DataFrame with 'IAT' column

   :returns: IAT regularity metric score (0-1)


