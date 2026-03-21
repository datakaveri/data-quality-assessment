import json

import pandas as pd

from report.aggregate_structured import generate_final_report
from report.json_writer import write_report_outputs
from report.multifile_average_score import calculate_average_readiness
from report.pdf_writer import generate_pdf_from_json
from report.scoring_structured import compute_aggregate_score


def build_raw_report():
    return {
        "column_missing": {"city": 100.0},
        "column_missing_count": 1,
        "column_missing_percentage": 25.0,
        "number_of_columns": 4,
        "row_missing_count": 1,
        "row_missing_percentage": 25.0,
        "number_of_rows": 4,
        "exact_row_duplicates_count": 1,
        "exact_row_duplicates_percentage": 25.0,
        "region_coverage": 20.0,
        "region_column": ["district"],
        "low_variance_numeric_columns": ["constant_metric"],
        "percentage_low_variance_numeric_columns": 50.0,
        "number_of_numeric_columns": 2,
        "numeric_columns": ["constant_metric", "value"],
        "dominant_categorical_columns": ["status"],
        "percentage_dominant_categorical_columns": 100.0,
        "number_of_categorical_columns": 1,
        "categorical_columns": ["status"],
        "file_format": "valid",
        "date_column": ["event_date"],
        "timestamp_column": ["updated_at"],
        "number_of_date_columns": 1,
        "number_of_timestamp_columns": 1,
        "datetime_issues_percentage": 50.0,
        "date_or_timestamp_fields_found": ["event_date", "updated_at"],
        "date_or_timestamp_issues_percentage": 25.0,
        "documentation_found": False,
    }


def test_compute_aggregate_score_excludes_llm_metrics():
    df = pd.DataFrame(
        {
            "city": [None, "A", "B", "C"],
            "constant_metric": [1, 1, 1, 1],
            "value": [10, 20, 30, 40],
            "status": ["open", "open", "open", "open"],
        }
    )

    score = compute_aggregate_score(build_raw_report(), df)

    assert score["detailed_scores"] == {
        "column_missing": 11.25,
        "row_missing": 7.5,
        "exact_row_duplicates": 7.5,
        "numeric_variance": 2.5,
        "file_format_check": 10,
    }
    assert score["total_weights"] == 50
    assert score["total_score"] == 38.75
    assert score["total_percentage"] == 77.5
    assert "coverage_check" not in score["detailed_scores"]
    assert "categorical_variation" not in score["detailed_scores"]
    assert "uniform_encoding" not in score["detailed_scores"]
    assert "date_or_timestamp_fields_found" not in score["detailed_scores"]


def test_generate_final_report_excludes_llm_sections(tmp_path):
    raw_path = tmp_path / "raw.json"
    raw_report = build_raw_report()
    raw_report.update(
        {
            "total_weights": 50,
            "total_score": 38.75,
            "total_percentage": 77.5,
            "detailed_scores": {
                "column_missing": 11.25,
                "row_missing": 7.5,
                "exact_row_duplicates": 7.5,
                "numeric_variance": 2.5,
                "file_format_check": 10,
            },
        }
    )
    raw_path.write_text(json.dumps(raw_report), encoding="utf-8")

    final_report = generate_final_report(str(raw_path))
    section_names = [section["bucket"] for section in final_report]
    test_keys = [test["key"] for section in final_report for test in section["tests"]]

    assert section_names == [
        "Data Quality",
        "Data Variance and Correctness",
        "Standardisation",
    ]
    assert test_keys == [
        "column_missing",
        "row_missing",
        "exact_row_duplicates",
        "numeric_variance",
        "file_format_check",
    ]


def test_calculate_average_readiness_recomputes_filtered_score(tmp_path):
    report_one = build_raw_report()
    report_one.update(
        {
            "total_weights": 50,
            "total_score": 50.0,
            "total_percentage": 100.0,
            "detailed_scores": {
                "column_missing": 15.0,
                "row_missing": 10.0,
                "exact_row_duplicates": 10.0,
                "numeric_variance": 5.0,
                "file_format_check": 10.0,
            },
        }
    )
    report_two = build_raw_report()
    report_two.update(
        {
            "file_format": "invalid",
            "total_weights": 50,
            "total_score": 25.0,
            "total_percentage": 50.0,
            "detailed_scores": {
                "column_missing": 7.5,
                "row_missing": 5.0,
                "exact_row_duplicates": 5.0,
                "numeric_variance": 2.5,
                "file_format_check": 5.0,
            },
        }
    )
    first_path = tmp_path / "report_one.json"
    second_path = tmp_path / "report_two.json"
    first_path.write_text(json.dumps(report_one), encoding="utf-8")
    second_path.write_text(json.dumps(report_two), encoding="utf-8")

    average_report, average_percentage = calculate_average_readiness(
        [str(first_path), str(second_path)]
    )

    assert average_report["detailed_scores"] == {
        "column_missing": 11.25,
        "row_missing": 7.5,
        "exact_row_duplicates": 7.5,
        "numeric_variance": 3.75,
        "file_format_check": 7.5,
    }
    assert average_report["total_weights"] == 50
    assert average_report["total_score"] == 37.5
    assert average_report["total_percentage"] == 75.0
    assert average_percentage == 75.0


def test_raw_json_keeps_llm_fields_but_filters_score_block(tmp_path):
    output_dir = tmp_path / "reports"
    final_score = {
        "total_weights": 50,
        "total_score": 38.75,
        "total_percentage": 77.5,
        "detailed_scores": {
            "column_missing": 11.25,
            "row_missing": 7.5,
            "exact_row_duplicates": 7.5,
            "numeric_variance": 2.5,
            "file_format_check": 10,
        },
    }

    write_report_outputs(final_score, str(output_dir), "dataset", build_raw_report())

    report_path = output_dir / "dataset_raw_readiness_report.json"
    written_report = json.loads(report_path.read_text(encoding="utf-8"))

    assert written_report["region_coverage"] == 20.0
    assert written_report["date_or_timestamp_fields_found"] == ["event_date", "updated_at"]
    assert written_report["detailed_scores"] == final_score["detailed_scores"]
    assert written_report["total_percentage"] == 77.5
    assert "coverage_check" not in written_report["detailed_scores"]


def test_generate_pdf_from_filtered_final_json(tmp_path):
    final_report = [
        {
            "bucket": "Data Quality",
            "weight": 35,
            "tests": [
                {
                    "id": "1.1",
                    "key": "column_missing",
                    "title": "Column-wise Missing",
                    "note": "All columns are sufficiently populated.",
                    "score": 15.0,
                    "max_score": 15,
                }
            ],
        },
        {
            "bucket": "Standardisation",
            "weight": 10,
            "tests": [
                {
                    "id": "3.1",
                    "key": "file_format_check",
                    "title": "File Format Check",
                    "note": "File format meets all requirements.",
                    "score": 10.0,
                    "max_score": 10,
                }
            ],
        },
    ]
    final_json_path = tmp_path / "final.json"
    pdf_path = tmp_path / "report.pdf"
    final_json_path.write_text(json.dumps(final_report), encoding="utf-8")

    keys = [test["key"] for section in final_report for test in section["tests"]]
    assert "coverage_check" not in keys
    assert "categorical_variation" not in keys
    assert "uniform_encoding" not in keys
    assert "date_or_timestamp_fields_found" not in keys

    generate_pdf_from_json(
        str(final_json_path),
        str(pdf_path),
        "dataset-id",
        50.0,
        str(tmp_path),
        "Dataset Name",
        sample_size=10,
    )

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0
