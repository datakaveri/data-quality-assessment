import json
from report.scoring_structured import RETAINED_STRUCTURED_METRIC_WEIGHTS

def calculate_average_readiness(reports):
    average_keys = {
        'column_missing_percentage',
        'row_missing_percentage',
        'exact_row_duplicates_percentage',
        'region_coverage',
        'percentage_low_variance_numeric_columns',
        'percentage_dominant_categorical_columns',
        'datetime_issues_percentage',
        'date_or_timestamp_issues_percentage',
    }
    sum_keys = {
        'column_missing_count',
        'number_of_columns',
        'row_missing_count',
        'number_of_rows',
        'exact_row_duplicates_count',
        'number_of_numeric_columns',
        'number_of_categorical_columns',
        'number_of_date_columns',
        'number_of_timestamp_columns',
    }
    list_keys = {
        'region_column',
        'low_variance_numeric_columns',
        'numeric_columns',
        'dominant_categorical_columns',
        'categorical_columns',
        'date_column',
        'timestamp_column',
        'date_or_timestamp_fields_found',
    }
    dict_keys = {'column_missing'}
    sentinel_defaults = {
        'region_column': "No region column found",
        'low_variance_numeric_columns': 'None',
        'numeric_columns': 'None',
        'dominant_categorical_columns': 'None',
        'categorical_columns': 'None',
        'date_column': 'None',
        'timestamp_column': 'None',
        'date_or_timestamp_fields_found': 'None',
    }
    averaged_detailed_scores = {metric: 0 for metric in RETAINED_STRUCTURED_METRIC_WEIGHTS}
    average_report = {}
    count = 0

    for report_path in reports:
        with open(report_path, "r") as f:
            report = json.load(f)

        if not average_report:
            average_report = {}

        for key, value in report.items():
            if key == 'detailed_scores' and isinstance(value, dict):
                for metric in averaged_detailed_scores:
                    averaged_detailed_scores[metric] += value.get(metric, 0)
            elif key in average_keys and isinstance(value, (int, float)):
                average_report[key] = average_report.get(key, 0) + value
            elif key in sum_keys and isinstance(value, (int, float)):
                average_report[key] = average_report.get(key, 0) + value
            elif key in dict_keys and isinstance(value, dict):
                merged = average_report.get(key, {}).copy()
                merged.update(value)
                average_report[key] = merged
            elif key in list_keys:
                items = value if isinstance(value, list) else []
                merged = set(average_report.get(key, []))
                merged.update(items)
                average_report[key] = sorted(merged)
            elif key == 'file_format':
                if average_report.get(key) == 'invalid' or value == 'invalid':
                    average_report[key] = 'invalid'
                else:
                    average_report[key] = value
            elif key == 'documentation_found':
                average_report[key] = average_report.get(key, False) or bool(value)
            elif key not in {'total_weights', 'total_score', 'total_percentage'}:
                average_report[key] = value
        count += 1

    for key in average_keys:
        if key in average_report and count > 0:
            average_report[key] = average_report[key] / count

    for key, default in sentinel_defaults.items():
        if key not in average_report or not average_report[key]:
            average_report[key] = default

    if 'column_missing' not in average_report:
        average_report['column_missing'] = {}

    if count > 0:
        averaged_detailed_scores = {
            metric: round(score / count, 2)
            for metric, score in averaged_detailed_scores.items()
        }
    total_weights = sum(RETAINED_STRUCTURED_METRIC_WEIGHTS.values())
    total_score = round(sum(averaged_detailed_scores.values()), 2)
    total_percentage = round(total_score / total_weights * 100, 2) if total_weights > 0 else 0

    average_report['detailed_scores'] = averaged_detailed_scores
    average_report['total_weights'] = total_weights
    average_report['total_score'] = total_score
    average_report['total_percentage'] = total_percentage
    average_percentage = total_percentage
    return average_report, average_percentage
