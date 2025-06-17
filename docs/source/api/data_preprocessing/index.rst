data_preprocessing
==================

.. py:module:: data_preprocessing


Functions
---------

.. autoapisummary::

   data_preprocessing.preProcess
   data_preprocessing.preProcess


Module Contents
---------------

.. py:function:: preProcess(df, input1, input2)

   Preprocesses the dataframe by converting observation time to datetime,
   sorting by uniqueID and time, and calculating inter-arrival times (IAT).

   :param df: DataFrame containing observationDateTime and uniqueID columns
   :param input1: Column name for unique identifier
   :param input2: Column name for datetime

   :returns: Processed DataFrame with IAT column.


.. py:function:: preProcess(df, input1, input2)

   Preprocesses the dataframe by converting observation time to datetime,
   sorting by uniqueID and time, and calculating inter-arrival times (IAT).

   :param df: DataFrame containing observationDateTime and uniqueID columns
   :param input1: Column name for unique identifier
   :param input2: Column name for datetime

   :returns: Processed DataFrame with IAT column


