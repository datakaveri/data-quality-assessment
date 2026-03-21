from structured_metrics.quality import *
from structured_metrics.relevance_completeness import *
from structured_metrics.variance_correctness import *
from structured_metrics.standardization import *
from structured_metrics.model_ingestible import *
from structured_metrics.regular_refresh import *
from structured_metrics.documentation import *
import json 
import logging 

def log_and_call(func, *args, **kwargs):
    logging.info(f"Calling function: {func.__name__}")
    return func(*args, **kwargs)


def generate_raw_report(df, data_file_path, imputed_columns=None):
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
    report.update(log_and_call(check_column_missing, df))
    report.update(log_and_call(check_row_missing, df))
    report.update(log_and_call(check_row_duplicates, df))
    report.update(log_and_call(check_coverage_region, df, imputed_columns))
    report.update(log_and_call(check_numeric_variance, df))
    report.update(log_and_call(check_categorical_variation, df, imputed_columns))
    report.update(log_and_call(check_file_format, data_file_path))
    report.update(log_and_call(check_date_and_timestamp_format, df, imputed_columns))
    report.update(log_and_call(check_date_or_timestamp_fields, df, imputed_columns))
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
            "column_missing": 15,
            "row_missing": 10,
            "exact_row_duplicates": 10,
            "numeric_variance": 5,
            "file_format_check": 10,
        }
        
        return {
            "column_missing": 
            f"All {int(readiness_metrics_raw['number_of_columns'])} columns have at least 70% of their data filled" if readiness_metrics_raw["detailed_scores"]["column_missing"] == max_scores["column_missing"] 
            else 
            f"{readiness_metrics_raw['number_of_columns'] - readiness_metrics_raw['column_missing_count']} out of {readiness_metrics_raw['number_of_columns']} columns have at least 70% of their data filled",

            "row_missing": 
            f"All {int(readiness_metrics_raw['number_of_rows'])} rows have at least 50% of fields populated." if readiness_metrics_raw["detailed_scores"]["row_missing"] == max_scores["row_missing"] 
            else 
            f"{readiness_metrics_raw['number_of_rows'] - readiness_metrics_raw['row_missing_count']} out of {readiness_metrics_raw['number_of_rows']} rows ({round(100 - readiness_metrics_raw['row_missing_percentage'], 1)}%) have at least 50% of fields populated.",

            "exact_row_duplicates": 
            f"100% of rows are unique with no duplicates detected." if readiness_metrics_raw["detailed_scores"]["exact_row_duplicates"] == max_scores["exact_row_duplicates"] 
            else 
            f"{round(100 - readiness_metrics_raw['exact_row_duplicates_percentage'], 1)}% of rows are unique, with {readiness_metrics_raw['exact_row_duplicates_count']} duplicate rows identified.",

            "numeric_variance": 
            "No numeric columns found." if readiness_metrics_raw["number_of_numeric_columns"] == 0 
            else (
                f"All {readiness_metrics_raw['number_of_numeric_columns']} numeric column(s) show sufficient statistical variation." 
                if readiness_metrics_raw["detailed_scores"]["numeric_variance"] == max_scores["numeric_variance"] 
                else 
                f"{readiness_metrics_raw['number_of_numeric_columns'] - len(readiness_metrics_raw['low_variance_numeric_columns'])} out of {readiness_metrics_raw['number_of_numeric_columns']} numeric columns show sufficient statistical variation."
            ),

            "file_format_check": 
            "File format meets all requirements." if readiness_metrics_raw["detailed_scores"]["file_format_check"] == max_scores["file_format_check"] 
            else 
            "File format provides opportunity for conversion to the required format.",
        }

    notes = get_notes()

    readiness_report = [
        {
            "bucket": "Data Quality",
            "weight": 35,
            "tests": [
                {
                    "id": "1.1",
                    "key": "column_missing",
                    "title": "Column-wise Missing",
                    "note": notes["column_missing"],
                    "score": detailed_scores["column_missing"],
                    "max_score": 15
                },
                {
                    "id": "1.2",
                    "key": "row_missing",
                    "title": "Row-wise Missing",
                    "note": notes["row_missing"],
                    "score": detailed_scores["row_missing"],
                    "max_score": 10
                },
                {
                    "id": "1.3",
                    "key": "exact_row_duplicates",
                    "title": "Row Duplicates",
                    "note": notes["exact_row_duplicates"],
                    "score": detailed_scores["exact_row_duplicates"],
                    "max_score": 10
                },
            ]
        },
        {
            "bucket": "Data Variance and Correctness",
            "weight": 0 if notes["numeric_variance"] == "No numeric columns found." else 5,
            "tests": [
                {
                    "id": "2.1",
                    "key": "numeric_variance",
                    "title": "Numeric Variance",
                    "note": notes["numeric_variance"],
                    "score": detailed_scores["numeric_variance"],
                    "max_score": 0 if notes["numeric_variance"] == "No numeric columns found." else 5
                }
            ]
        },
        {
            "bucket": "Standardisation",
            "weight": 10,
            "tests": [
                {
                    "id": "3.1",
                    "key": "file_format_check",
                    "title": "File Format Check",
                    "note": notes["file_format_check"],
                    "score": detailed_scores["file_format_check"],
                    "max_score": 10
                }
            ]
        }
    ]

    return readiness_report
