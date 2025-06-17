DQReportGenerator
=================

.. py:module:: DQReportGenerator


Attributes
----------

.. autoapisummary::

   DQReportGenerator.configFile
   DQReportGenerator.numPackets
   DQReportGenerator.schemaProvision
   DQReportGenerator.schemaInputValidity
   DQReportGenerator.schemaFile
   DQReportGenerator.dfClean
   DQReportGenerator.regularityMetricScore
   DQReportGenerator.outliersMetricScore
   DQReportGenerator.dupeMetricScore
   DQReportGenerator.compMetricScore
   DQReportGenerator.formatMetricScore
   DQReportGenerator.addnlAttrMetricScore
   DQReportGenerator.avgDataQualityScore
   DQReportGenerator.avgDataQualityPercent
   DQReportGenerator.WIDTH
   DQReportGenerator.HEIGHT
   DQReportGenerator.fileNameNoExt
   DQReportGenerator.outputParamFV
   DQReportGenerator.myJSON
   DQReportGenerator.filename
   DQReportGenerator.jsonpath


Classes
-------

.. autoapisummary::

   DQReportGenerator.pdf


Functions
---------

.. autoapisummary::

   DQReportGenerator.create_title_card
   DQReportGenerator.create_heading
   DQReportGenerator.create_analytics_report_schema


Module Contents
---------------

.. py:data:: configFile

.. py:data:: numPackets

.. py:data:: schemaProvision

.. py:data:: schemaInputValidity
   :value: 0


.. py:data:: schemaFile

.. py:data:: dfClean

.. py:data:: regularityMetricScore

.. py:data:: outliersMetricScore

.. py:data:: dupeMetricScore

.. py:data:: compMetricScore

.. py:data:: formatMetricScore

.. py:data:: addnlAttrMetricScore

.. py:data:: avgDataQualityScore

.. py:data:: avgDataQualityPercent

.. py:data:: WIDTH
   :value: 210


.. py:data:: HEIGHT
   :value: 297


.. py:data:: fileNameNoExt

.. py:function:: create_title_card(pdf)

.. py:function:: create_heading(title, pdf)

.. py:class:: pdf(orientation='P', unit='mm', format='A4')

   Bases: :py:obj:`fpdf.FPDF`


   PDF Generation class


   .. py:method:: add_page(same=True, orientation='')

      Start a new page



   .. py:method:: footer()

      Footer to be implemented in your own inherited class



.. py:function:: create_analytics_report_schema(filename=f'{fileNameNoExt}_DQReport.pdf')

.. py:data:: outputParamFV

.. py:data:: myJSON

.. py:data:: filename

.. py:data:: jsonpath

