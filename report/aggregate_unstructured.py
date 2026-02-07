from unstructured_metrics.file_duplicates import *
from unstructured_metrics.file_type_consistency import *
from unstructured_metrics.documentation import *
from unstructured_metrics.file_format_check import *
from unstructured_metrics.file_openability import *
from unstructured_metrics.model_ingestible import *
from unstructured_metrics.coverage import *
from unstructured_metrics.timestamps_presence import *

import json 
import logging 

def log_and_call(func, *args, **kwargs):
    logging.info(f"Calling function: {func.__name__}")
    return func(*args, **kwargs)


def generate_raw_report(data_file_path, imputed_roles=None):
    """
    Generate a raw data quality report from a given dataframe, descriptor path, and data directory.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to generate the report from.
    descriptor_path : str
        The path to the descriptor file.
    data_directory : str
        The path to the data directory.

    Returns
    -------
    dict
        A dictionary containing the raw data quality metrics.
    """
    report = {}
    report.update(log_and_call(check_file_duplicates, data_file_path))
    report.update(log_and_call(check_type_uniformity, data_file_path))
    report.update(log_and_call(check_file_openability, data_file_path))
    report.update(log_and_call(check_label_presence, data_file_path))
    report.update(log_and_call(check_timestamp_presence, imputed_roles))
    report.update(log_and_call(region_coverage, imputed_roles))
    report.update(log_and_call(check_file_format, data_file_path))
    report.update(log_and_call(check_documentation_presence, data_file_path))
    return report

def generate_final_report(readiness_metrics_json_path):
    """
    Generate a final data quality report from a given raw data quality report.

    Parameters
    ----------
    readiness_metrics_json_path : str
        The path to the raw data quality report JSON file.

    Returns
    -------
    list
        A list of dictionaries containing the final data quality report.
    """
    with open(readiness_metrics_json_path, "r") as f:
        readiness_metrics_raw = json.load(f)
    detailed_scores = readiness_metrics_raw["detailed_scores"]

    # Notes are constructed using raw metrics from the report
    def get_notes():
        """
        Construct a dictionary of notes for the final readiness report based on raw report metrics.

        Notes are strings that provide a brief description of the result of a given test. For example, if the test is 
        checking column-wise missing, the note will contain the number of columns that exceeded the missing threshold.

        Returns
        -------
        dict
            A dictionary with keys matching the raw report metrics and values as strings containing the notes.
        """
        max_scores = {
            "consistency": 15,
            "duplicate_files": 15,
            "region_coverage": 10,
            "file_openability": 15,
            "file_format_check": 15,
            "annotation_presence": 10,
            "timestamps_presence": 10,
            "documentation_presence": 10,
        }
        
        return {
            "consistency": 
            "All file types are uniform." if readiness_metrics_raw["consistency"] else "File types are inconsistent.",

            "duplicate_files": 
            f"{100 - int(readiness_metrics_raw['duplicate_percentage'])}% of files are duplicates.",

            "region_coverage": 
            f"{readiness_metrics_raw['region_coverage']}% coverage achieved for region metadata." if readiness_metrics_raw['region_coverage'] else 'None',

            "file_openability": 
                "All sampled files are openable without errors." if readiness_metrics_raw["detailed_scores"]["file_openability"] == max_scores["file_openability"]
                else f"{readiness_metrics_raw['file_openable_percentage']}% of sampled files are openable without errors.",
            
            "file_format_check": 
            f"{readiness_metrics_raw['valid_format_percentage']}% of files are in a standard format.",
            
            "annotation_presence": 
            "Annotations are present." if readiness_metrics_raw["annotation_presence"] 
            else 'None',

            "timestamps_presence": 
            "Timestamps are present." if readiness_metrics_raw["timestamps_presence"] 
            else 'None',

            "documentation_presence": (
                "Documentation includes comprehensive data dictionary files." if readiness_metrics_raw["detailed_scores"]["documentation_presence"] == max_scores["documentation_presence"] 
                else 
                "None"
            )
        }

    notes = get_notes()

    readiness_report = [
        {
            "bucket": "Data Quality",
            "weight": 30,
            "tests": [
                {
                    "id": "1.1",
                    "key": "consistency",
                    "title": "Consistency",
                    "note": notes["consistency"],
                    "score": detailed_scores["consistency"],
                    "max_score": 15
                },
                {
                    "id": "1.2",
                    "key": "duplicate_files",
                    "title": "Duplicate Files",
                    "note": notes["duplicate_files"],
                    "score": detailed_scores["duplicate_files"],
                    "max_score": 15
                }
            ]
        },
        {
            "bucket": "Data Relevance and Completeness",
            "weight": 0 if notes["region_coverage"] == "None" or detailed_scores["region_coverage"] == False else 10,
            "tests": [
                {
                    "id": "2.1",
                    "key": "region_coverage",
                    "title": "Coverage Check",
                    "note": notes["region_coverage"],
                    "score": detailed_scores["region_coverage"],
                    "max_score": 0 if notes["region_coverage"] == "None" or detailed_scores["region_coverage"] == False else 10
                },
            ]
        },
        {
            "bucket": "Standardisation",
            "weight": 30,
            "tests": [
                {
                    "id": "3.1",
                    "key": "file_format_check",
                    "title": "File Format Check",
                    "note": notes["file_format_check"],
                    "score": detailed_scores["file_format_check"],
                    "max_score": 15
                },
                {
                    "id": "3.2",
                    "key": "file_openability",
                    "title": "File Openability",
                    "note": notes["file_openability"],
                    "score": detailed_scores["file_openability"],
                    "max_score": 15
                }
            ]
        },
        {
            "bucket": "Model Ingestible Data",
            "weight": 0 if notes["annotation_presence"] == "None" or detailed_scores["annotation_presence"] == 0 else 10,
            "tests": [
                {
                    "id": "4.1",
                    "key": "annotation_presence",
                    "title": "Annotation Presence",
                    "note": notes["annotation_presence"],
                    "score": detailed_scores["annotation_presence"],
                    "max_score": 0 if notes["annotation_presence"] == "None" or detailed_scores["annotation_presence"] == 0 else 10
                }
            ]
        },
        {
            "bucket": "Regular Refresh",
            "weight": 0 if notes["timestamps_presence"] == "None" or detailed_scores["timestamps_presence"] == 0 else 10,
            "tests": [
                {
                    "id": "5.1",
                    "key": "timestamps_presence",
                    "title": "Timestamps Presence",
                    "note": notes["timestamps_presence"],
                    "score": detailed_scores["timestamps_presence"],
                    "max_score": 0 if notes["timestamps_presence"] == "None" or detailed_scores["timestamps_presence"] == 0 else 10
                }
            ]
        # },
        # {
        #     "bucket": "Documentation",
        #     "weight": 0 if notes["documentation_presence"] == "None" or detailed_scores["documentation_presence"] == 0 else 15,
        #     "tests": [
        #         {
        #             "id": "6.1",
        #             "key": "documentation_presence",
        #             "title": "Documentation Presence",
        #             "note": notes["documentation_presence"],
        #             "score": detailed_scores["documentation_presence"],
        #             "max_score": 10
        #         }
        #     ]
        # }
        }
    ]

    return readiness_report