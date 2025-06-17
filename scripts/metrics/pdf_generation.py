import os
from fpdf import FPDF
from fpdf import *

WIDTH = 210
HEIGHT = 297

def create_title_card(pdf, datasetName, URL, numPackets, startTime, endTime):
    """
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
    """
    pdf.image("../plots/pretty/iudx.png", 10, 5, 35)
    pdf.set_font("times", "b", 22)
    pdf.set_x(60)
    pdf.write(5, "Data Quality Report")
    pdf.ln(10)
    pdf.set_font("times", "", 12)
    pdf.ln(7)
    pdf.write(5, "Dataset: ")
    pdf.set_text_color(r=0, g=0, b=105)
    pdf.cell(10, 5, f"{datasetName}", link=URL, align="L")
    pdf.set_text_color(r=0, g=0, b=0)
    pdf.ln(7)
    pdf.cell(
        15,
        5,
        f"Number of Data Packets: {numPackets}    |    Start Time: {startTime}    |   End Time: {endTime}",
    )
    pdf.line(11, 41, 200, 41)

def create_heading(title, pdf):
    pdf.set_font("times", "b", 18)
    pdf.ln(10)
    pdf.write(5, title)
    pdf.ln(5)
    pdf.set_font("times", "", 12)

class PDFReport(FPDF):
    def add_page(self, same=True, orientation=""):
        FPDF.add_page(self, same=same, orientation=orientation)

    def footer(self):
        # Page number with condition isCover
        self.set_y(-15)  # Position at 1.5 cm from bottom
        self.set_font("times", "I", 8)
        self.cell(0, 10, "Page  " + str(self.page_no) + "  |  {nb}", 0, 0, "C")

def create_analytics_report_schema(
    filename,
    datasetName,
    URL,
    numPackets,
    startTime,
    endTime,
    regularityMetricScore,
    outliersMetricScore,
    dupeMetricScore,
    compMetricScore,
    formatMetricScore,
    addnlAttrMetricScore,
    avgDataQualityScore,
    avgDataQualityPercent,
    input1,
    input2,
    dupeCount
):
    """
    Generate a comprehensive data quality PDF report.
    
    Args:
        filename: Name of the output PDF file
        datasetName: Name of the dataset
        URL: URL link for the dataset
        numPackets: Number of data packets
        startTime: Start time of the dataset
        endTime: End time of the dataset
        regularityMetricScore: IAT regularity metric score
        outliersMetricScore: IAT outliers metric score
        dupeMetricScore: Duplicates metric score
        compMetricScore: Completeness metric score
        formatMetricScore: Format adherence metric score
        addnlAttrMetricScore: Additional attributes metric score
        avgDataQualityScore: Average data quality score
        avgDataQualityPercent: Average data quality percentage
        input1: First attribute for deduplication
        input2: Second attribute for deduplication
        dupeCount: Number of duplicate packets found
    """
    
    pdf = FPDF()  # A4 (210 by 297 mm)

    # First Page
    pdf.add_page()
    # Adding the banner/letterhead
    create_title_card(pdf, datasetName, URL, numPackets, startTime, endTime)
    create_heading("Overview", pdf)
    pdf.ln(5)

    # Creating table of metric scores overview
    data = [
        ["Metric", "Score", "Bar"],
        ["Inter-Arrival Time Regularity", f"{regularityMetricScore}", ""],
        ["Inter-Arrival Time Outliers", f"{outliersMetricScore}", ""],
        ["Duplicate Presence", f"{dupeMetricScore}", ""],
        ["Adherence to Attribute Format", f"{formatMetricScore}", ""],
        ["Absence of Unknown Attributes", f"{addnlAttrMetricScore}", ""],
        ["Adherence to Mandatory Attributes", f"{compMetricScore}", ""],
    ]

    # Text height is the same as current font size
    # Effective page width, or just epw
    epw = pdf.w - 2 * pdf.l_margin

    # Set column width to 1/3 of effective page width to distribute content evenly across table and page
    col_width = epw / 3
    th = pdf.font_size

    # Logic included to bold only titles
    for row in data:
        for index, datum in enumerate(row):
            if datum == "Metric" or datum == "Score" or datum == "Bar":
                pdf.set_font("times", "b", 15)
                if index == 1:
                    pdf.cell(col_width / 2.7, 4 * th, str(datum), border=1, align="C")
                elif index == 2:
                    pdf.cell(col_width * 1.5, 4 * th, str(datum), border=1, align="C")
                else:
                    pdf.cell(col_width + 10, 4 * th, str(datum), border=1, align="C")
            elif datum != "Metric" or datum != "Score" or datum != "Bar":
                pdf.set_font("times", "", 12)
                if index == 1:
                    pdf.cell(col_width / 2.7, 4 * th, str(datum), border=1, align="C")
                elif index == 2:
                    pdf.cell(col_width * 1.5, 4 * th, str(datum), border=1, align="C")
                else:
                    pdf.cell(col_width + 10, 4 * th, str(datum), border=1, align="C")
        pdf.ln(4 * th)

    # Adding bars to the table
    pdf.image("../plots/bars/regularitybar.png", 107, 70, 95)
    pdf.image("../plots/bars/outliersbar.png", 107, 88, 95)
    pdf.image("../plots/bars/dupebar.png", 107, 104, 95)
    pdf.image("../plots/bars/formatbar.png", 107, 121, 95)
    pdf.image("../plots/bars/addnlbar.png", 107, 138, 95)
    pdf.image("../plots/bars/compbar.png", 107, 155, 95)

    pdf.ln(10)
    pdf.write(
        5,
        "The Overall Data Quality Score of the dataset, computed by calculating an average of the above scores is:",
    )
    pdf.ln(10)
    pdf.set_font("times", "b", 12)
    pdf.write(5, f"{avgDataQualityScore}/1.00 or {avgDataQualityPercent}%")
    pdf.set_font("times", "", 12)
    pdf.ln(35)

    # Radar chart
    pdf.image("../plots/radarPlot.png", 110, 195, 95)

    pdf.write(5, "This data quality assessment report shows the score for")
    pdf.ln(5)
    pdf.write(5, "six metrics that contribute to data quality.")
    pdf.ln(10)
    pdf.write(5, "The chart on the right shows an overview of the data")
    pdf.ln(5)
    pdf.write(5, "quality of the dataset.")
    pdf.ln(10)
    pdf.write(5, "In the following pages you can find a detailed description")
    pdf.ln(5)
    pdf.write(5, "and breakdown of each of these metrics.")

    # Second Page
    pdf.add_page()
    pdf.ln(5)
    create_heading("Inter-Arrival Time Regularity", pdf)
    pdf.ln(5)
    pdf.write(
        5,
        "Inter-arrival time is defined as the time elapsed after the receipt of a data packet and until the receipt of the next packet. For sensor data, this is an important factor to evaluate as sensors are often configured to send data at specific time intervals.",
    )
    pdf.ln(5)
    pdf.image("../plots/donuts/regularityMetricScorePiePlot.png", x=150, y=-5, w=60)
    pdf.ln(10)
    pdf.write(
        5,
        "In order to compute this metric we analyse the deviation of each inter-arrival time from the mode. The assumption here is if most of the sensors are operating nominally most of the time, then the mode of the inter-arrival times will represent the expected nominal behaviour of the sensors. To compute this deviation, we define:",
    )
    pdf.ln(5)
    pdf.image("../plots/equations/RAE_regularityMetric.png", x=65, y=80, w=80)
    pdf.ln(20)
    pdf.write(
        5,
        "Here, xi is the inter-arrival time, and x is the mode of the inter-arrival time. We consider an RAE value of 0.5 to be the crossover point between good and poor values of inter-arrival time, i.e. RAE > 0.5 is poor. We also want to penalise the score proportionately to the RAE value, meaning the greater the RAE value, the greater the penalty. RAE is thus bound as RAE belongs to [0, inf).",
    )
    pdf.ln(5)
    pdf.write(5, "The metric computation can also be represented as an equation:")
    pdf.ln(5)
    pdf.image("../plots/equations/regularityMetric.png", x=60, y=130, w=80)
    pdf.ln(40)
    pdf.write(
        5,
        'This represents the "badness" of the inter-arrival time when compared to the modal value. The further the inter-arrival time is from the mode, the greater the penalty contribution to the regularity score for that inter-arrival time. A value of 0.5 for RAE is chosen as the crossover point between "goodness" and "badness" of inter-arrival time as it represents a window of values corresponding to:',
    )
    pdf.image("../plots/equations/mode_regularityMetric.png", x=90, y=190, w=20)
    pdf.ln(20)
    pdf.write(
        5,
        "A higher IAT Regularity score indicates lower dispersion of IAT values around the mode, and vice versa. A higher score indicates that there is a higher clustering of IAT values close to the mode of the sensor. This regularity is particularly important for time-critical applications where a consistent and predictable arrival pattern is desired. By evaluating the IAT Regularity metric, researchers can gain insights into the reliability and efficiency of the data transmission process in IoT networks, contributing to the optimization of various IoT applications and services.",
    )
    pdf.ln(10)

    # Fourth Page
    pdf.add_page()
    pdf.ln(5)
    create_heading("Inter-Arrival Time Outliers", pdf)
    pdf.image("../plots/donuts/outliersMetricScorePiePlot.png", x=150, y=-5, w=60)
    pdf.ln(5)
    pdf.write(
        5,
        "The outlier metric of the inter-arrival time is an evaluation of the number of IAT values that show a significant deviation from the expected behaviour.",
    )
    pdf.ln(10)
    pdf.write(
        5,
        "There are multiple ways to identify outliers in a dataset, and the choice of method is dependent on the independent characteristics of the dataset. In our case, we apply the modified z-score method proposed by Iglewiscz and Hoaglin.",
    )
    pdf.ln(5)
    pdf.write(5, "Let the Median Absolute Deviation of the data be defined as: ")
    pdf.ln(10)
    pdf.image("../plots/equations/median_OutliersMetric.png", x=60, y=75, w=55)
    pdf.ln(20)
    pdf.write(
        5,
        "where xi is the observation for which the MAD is being computed and x is the mode of the data. We use the mode in place of the median as used by Iglewiscz and Hoaglin because we want to evaluate the deviation of the inter-arrival times from the mode, and we consider the mode to represent the expected behaviour of the dataset. Then the modified Z-score Mi is: ",
    )
    pdf.ln(10)
    pdf.image("../plots/equations/modZscore_OutliersMetric.png", x=65, y=120, w=45)
    pdf.ln(20)
    pdf.write(
        5,
        "Here, Iglewiscz and Hoaglin suggest that observations with |Mi| > 3.5 be classified as outliers, with variations to this cut-off value depending on the distribution of x. For our purposes, we will use this value to label inter-arrival time values as outliers. The outliers for this dataset are shown in the plot below.",
    )
    pdf.ln(10)
    pdf.image("../plots/outliersPlot.png", x=15, y=155, w=WIDTH - 50)

    # Fifth Page
    pdf.add_page()
    create_heading("Duplicate Detection", pdf)
    pdf.image("../plots/donuts/dupeMetricScorePiePlot.png", x=150, y=-5, w=60)
    pdf.ln(10)
    pdf.write(
        5,
        "This metric conveys how many duplicate data points are present in the dataset.",
    )
    pdf.ln(10)
    pdf.write(
        5,
        "The duplicates in a dataset are identified using the timestamp and any one unique identifier for each data packet. For example: AQM Sensor ID, Vehicle ID, etc. may be used as unique identifiers for a dataset.",
    )
    pdf.ln(5)
    pdf.write(
        5,
        "If any unique identifier sends two data packets with the same timestamp, then one of the two data packets is counted as a duplicate. This is because it is assumed that any one device or sensor may not send two data packets with a single timestamp.",
    )
    pdf.ln(10)
    pdf.write(5, "For this dataset, the attributes chosen for deduplication are: ")
    pdf.ln(10)
    pdf.set_font("times", "b", 12)
    pdf.write(5, f"{input1}")
    pdf.ln(5)
    pdf.write(5, f"{input2}")
    pdf.ln(10)
    pdf.set_font("times", "", 12)
    pdf.write(
        5,
        f"Using these attributes, {dupeCount} duplicate data packets have been identified in the dataset.",
    )
    pdf.ln(5)
    pdf.write(
        5,
        "This metric is calculated on a score from 0 to 1, where a score of 0 indicates that all the data packets are duplicates and a score of 1 indicates that none of the data packets are duplicates.",
    )
    pdf.ln(5)
    pdf.write(
        5,
        "The chart below shows the number of data packets before and after deduplication on a per unique ID basis. If a unique ID is not represented in the chart, it means that there were no duplicate values received from that unique ID.",
    )
    pdf.ln(10)
    pdf.image("../plots/dupePlotID.png", x=20, y=140, w=170)

    # Sixth Page
    pdf.add_page()
    create_heading("Metrics for Schema Analysis", pdf)
    pdf.ln(10)
    pdf.write(
        5,
        "The remaining three metrics are an analysis of the metadata that is provided along with the dataset. This metadata is provided in the form of a schema, a document that delineates the different types of attributes, the data types of each attribute (integer, float, string, etc.) as well as the range of the observations under each attribute. This document also provides the mandatory attributes that the dataset must contain, as well as a list of all the expected attributes in the dataset.",
    )
    pdf.ln(10)

    create_heading("Attribute Format Adherence", pdf)
    pdf.image("../plots/donuts/formatMetricScorePiePlot.png", x=150, y=50, w=60)
    pdf.ln(10)
    pdf.write(
        5,
        "The attribute format metric checks whether the format of the data packets being evaluated matches the format defined in the data schema. The various possible formats include number, string, float, and object.",
    )
    pdf.ln(5)
    pdf.write(
        5,
        "The format adherence metric is computed using the json schema validation method. The count of errors is incremented when the data type of an evaluated data packet does not match the data type specified in the data schema.",
    )
    pdf.ln(5)
    pdf.write(
        5,
        "A higher score for the attribute format metric indicates a relatively lower proportion of data packets that contain attributes that do not adhere to the format defined in the schema, and a lower score for the attribute format metric indicates a relatively greater proportion of data packets with incorrect attribute formats.",
    )
    pdf.ln(10)

    create_heading("Absence of Unknown Attributes", pdf)
    pdf.image("../plots/donuts/addnlAttrMetricScorePiePlot.png", x=150, y=120, w=60)
    pdf.ln(10)
    pdf.write(
        5,
        "The unknown attributes A higher score for the attribute format metric indicates a relatively lower proportion of data packets that contain attributes that do not adhere to the format defined in the schema, and a lower score for the attribute format metric indicates a relatively greater proportion of data packets with incorrect attribute formats.metric computes the number of data packets with attributes that are present in the dataset but are not specified in the schema in any capacity.",
    )
    pdf.ln(5)
    pdf.write(
        5,
        "This metric is computed by validating the data against the schema. A higher score for this metric indicates a relatively lower proportion of data packets that contain attributes that are not present in the data schema and a lower score indicates a relatively greater proportion of data packets with unknown attributes.",
    )
    pdf.ln(5)
    pdf.write(
        5,
        "This metric represents the total number of unknown attributes in the dataset.",
    )
    pdf.ln(10)

    create_heading("Adherence to Mandatory Attributes", pdf)
    pdf.image("../plots/donuts/compMetricScorePiePlot.png", x=150, y=195, w=60)
    pdf.ln(10)
    pdf.write(
        5,
        "The mandatory attributes metric checks whether the list of mandatory attributes defined in the data schema are all present in the dataset. This validation is performed for each data packet in the dataset.",
    )
    pdf.ln(5)
    pdf.write(
        5,
        "A higher score for the mandatory attributes metric indicates that there is a relatively greater proportion of data packets with values present for all the mandatory attributes, and a lower score for the mandatory attributes metric indicates that there is a relatively lower proportion.",
    )
    pdf.ln(5)
    pdf.write(
        5,
        "This metric is an indicator of the completeness of the dataset. Null values received under mandatory attributes are also included in the count of the number of missing attributes.",
    )

    pdf.output("../outputReports/" + filename, "F")
    print(f"PDF report '{filename}' has been generated successfully!")
