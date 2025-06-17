test_preprocessing_new
======================

.. py:module:: test_preprocessing_new


Classes
-------

.. autoapisummary::

   test_preprocessing_new.TestPreProcessing


Functions
---------

.. autoapisummary::

   test_preprocessing_new.setup_imports


Module Contents
---------------

.. py:function:: setup_imports()

   Setup imports for the test file to work from any location


.. py:class:: TestPreProcessing

   .. py:method:: sample_config()

      Sample configuration dictionary for testing



   .. py:method:: sample_json_data()

      Sample JSON data for testing



   .. py:method:: sample_dataframe()

      Sample DataFrame for testing



   .. py:method:: test_readFile_success(sample_config, sample_json_data)

      Test successful file reading



   .. py:method:: test_timeRange(sample_dataframe)

      Test time range calculation



   .. py:method:: test_dropDupes(sample_dataframe, capsys)

      Test duplicate removal



   .. py:method:: test_outRemove(sample_dataframe)

      Test outlier removal using IQR method



   .. py:method:: test_dataStats(sample_dataframe)

      Test statistical calculations



   .. py:method:: test_radarChart(mock_radar)

      Test radar chart creation



   .. py:method:: test_dropDupes_no_duplicates(sample_dataframe)

      Test dropDupes with no duplicates



