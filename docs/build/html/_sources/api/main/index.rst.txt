main
====

.. py:module:: main


Functions
---------

.. autoapisummary::

   main.main


Module Contents
---------------

.. py:function:: main()

   Main function to execute data quality assessment pipeline.

   Orchestrates the complete data quality analysis workflow including:
   - Configuration file reading and data loading
   - Schema validation
   - Data preprocessing and outlier removal
   - Calculation of various data quality metrics
   - Generation of visualizations and reports

   The function interactively prompts user for configuration file name,
   schema validation preferences, and PDF report generation choices.

   :returns: Outputs JSON report file and optional PDF report to disk
   :rtype: None


