pdf_generation
==============

.. py:module:: pdf_generation


Attributes
----------

.. autoapisummary::

   pdf_generation.WIDTH
   pdf_generation.HEIGHT


Classes
-------

.. autoapisummary::

   pdf_generation.PDFReport


Functions
---------

.. autoapisummary::

   pdf_generation.create_title_card
   pdf_generation.create_heading
   pdf_generation.create_analytics_report_schema


Module Contents
---------------

.. py:data:: WIDTH
   :value: 210


.. py:data:: HEIGHT
   :value: 297


.. py:function:: create_title_card(pdf, datasetName, URL, numPackets, startTime, endTime)

       Creates the title card for the PDF report.

       Args:
           pdf: FPDF object to add content to.
           datasetName: Name of the dataset.
           URL: URL link to the dataset.
           numPackets: Number of data packets in the dataset.
           startTime: Start time of the data collection.
           endTime: End time of the data collection.

       Returns:
   <<<<<<< HEAD
           None, Modifies the PDF object in place.
   =======
           None. Modifies the PDF object in place.
   >>>>>>> 9c5f2989031ba54019bec835b7ecb3f5768f2dcf



.. py:function:: create_heading(title, pdf)

.. py:class:: PDFReport(orientation='P', unit='mm', format='A4')

   Bases: :py:obj:`fpdf.FPDF`


   PDF Generation class


   .. py:method:: add_page(same=True, orientation='')

      Start a new page



   .. py:method:: footer()

      Footer to be implemented in your own inherited class



.. py:function:: create_analytics_report_schema(filename, datasetName, URL, numPackets, startTime, endTime, regularityMetricScore, outliersMetricScore, dupeMetricScore, compMetricScore, formatMetricScore, addnlAttrMetricScore, avgDataQualityScore, avgDataQualityPercent, input1, input2, dupeCount)

   Generate a comprehensive data quality PDF report.

   :param filename: Name of the output PDF file
   :param datasetName: Name of the dataset
   :param URL: URL link for the dataset
   :param numPackets: Number of data packets
   :param startTime: Start time of the dataset
   :param endTime: End time of the dataset
   :param regularityMetricScore: IAT regularity metric score
   :param outliersMetricScore: IAT outliers metric score
   :param dupeMetricScore: Duplicates metric score
   :param compMetricScore: Completeness metric score
   :param formatMetricScore: Format adherence metric score
   :param addnlAttrMetricScore: Additional attributes metric score
   :param avgDataQualityScore: Average data quality score
   :param avgDataQualityPercent: Average data quality percentage
   :param input1: First attribute for deduplication
   :param input2: Second attribute for deduplication
   :param dupeCount: Number of duplicate packets found


