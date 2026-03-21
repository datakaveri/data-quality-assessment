RETAINED_STRUCTURED_METRIC_WEIGHTS = {
    "column_missing": 15,
    "row_missing": 10,
    "exact_row_duplicates": 10,
    "numeric_variance": 5,
    "file_format_check": 10,
}


def _get_total_rows(report_dict, df=None):
    if "number_of_rows" in report_dict:
        return report_dict["number_of_rows"]
    if df is not None:
        return len(df)
    return 0


def compute_aggregate_score(report_dict, df=None):
    """
    Computes the aggregate score from a dictionary of individual metrics.

    Parameters
    ----------
    report_dict : dict
        A dictionary of individual metrics, each with a score out of 100
    df : pandas.DataFrame
        The DataFrame containing the dataset

    Returns
    -------
    final_report : dict
        A dictionary with the total score and detailed scores for each metric
    """
    total_score = 0
    detailed_scores = {}

    weights = RETAINED_STRUCTURED_METRIC_WEIGHTS

    # 1. Column-wise Missing (score decreases as missing % increases)
    if "column_missing" in report_dict:
        cols = report_dict["column_missing"]
        if "column_missing_count" in report_dict and "number_of_columns" in report_dict:
            missing_cols = report_dict["column_missing_count"]
            total_cols = report_dict["number_of_columns"]
            avg_missing = (missing_cols / total_cols) * 100
            score = max(0, weights["column_missing"] * (1 - avg_missing / 100))
        else:
            score = weights["column_missing"]
        detailed_scores["column_missing"] = round(score, 2)
        total_score += score

    # 2. Row-wise Missing (proportion of rows with >50% missing)
    if "row_missing_count" in report_dict:
        affected_rows = report_dict["row_missing_count"]
        total_rows = _get_total_rows(report_dict, df)
        prop = affected_rows / total_rows if total_rows > 0 else 1
        score = max(0, weights["row_missing"] * (1 - prop))
        detailed_scores["row_missing"] = round(score, 2)
        total_score += score

    # 3. Exact Row Duplicates
    if "exact_row_duplicates_count" in report_dict:
        dupes = report_dict["exact_row_duplicates_count"]
        total_rows = _get_total_rows(report_dict, df)
        prop = dupes / total_rows if total_rows > 0 else 1
        score = max(0, weights["exact_row_duplicates"] * (1 - prop))
        detailed_scores["exact_row_duplicates"] = round(score, 2)
        total_score += score

    # 5. Numeric Variance (low-variance columns)
    if "low_variance_numeric_columns" in report_dict:
        percent = report_dict["percentage_low_variance_numeric_columns"]
        if percent > 0:
            score = max(0, weights["numeric_variance"] * (1 - percent / 100))
        else:
            score = weights["numeric_variance"]
        detailed_scores["numeric_variance"] = round(score, 2)
        total_score += score

    # 7. File Format Check (Boolean)
    if "file_format" in report_dict:
        score = weights["file_format_check"] if report_dict["file_format"] == "valid" else 0

        detailed_scores["file_format_check"] = score
        total_score += score
    
    # Output the results as a dictionary
    total_weights = sum(weights.values())
    final_report = {
        "total_weights": total_weights,
        "total_score": round(total_score, 2),
        "total_percentage": round(total_score / total_weights * 100, 2) if total_weights > 0 else 0,
        "detailed_scores": detailed_scores
    }
    
    return final_report
