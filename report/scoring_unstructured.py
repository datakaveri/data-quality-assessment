def compute_aggregate_score(report_dict):
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

    # Define metric weights (out of 100)
    weights = {
        "consistency": 15,
        "duplicate_files": 15,
        "region_coverage": 10 if report_dict.get("region_coverage") == True else 0,
        "file_openability": 15,
        "file_format_check": 15,
        "annotation_presence": 10 if report_dict.get("annotation_presence") == True else 0,
        "timestamps_presence": 10 if report_dict.get("timestamps_presence") == True else 0,
        "documentation_presence": 0,
    }

    # 1. File Type Consistency (binary scoring)
    if "consistency" in report_dict:
        if "consistency" in report_dict and report_dict["consistency"] != False:
            score = weights["consistency"]
        else:
            score = 0
        detailed_scores["consistency"] = round(score, 2)
        total_score += score

    # 2. Duplicate files (percentage of files that are duplicates)
    if "duplicate_percentage" in report_dict:
        duplicate_percentage = report_dict["duplicate_percentage"]
        prop = duplicate_percentage
        score = max(0, weights["duplicate_files"] * (1 - prop))
        detailed_scores["duplicate_files"] = round(score, 2)
        total_score += score

    # 3. Region Coverage
    if "region_coverage" in report_dict:
        if "region_coverage" in report_dict and report_dict["region_coverage"] == True:
            score = weights["region_coverage"]
        else:
            score = 0
        detailed_scores["region_coverage"] = round(score, 2)
        total_score += score

    # 4. File Openability (Proportion of files that open without any errors)
    if "file_openable_percentage" in report_dict:
        prop = report_dict["file_not_openable_percentage"]
        score = max(0, weights["file_openability"] * (1 - prop / 100))
        detailed_scores["file_openability"] = round(score, 2)
        total_score += score

    # 5. File Format Check
    if "valid_format_percentage" in report_dict:
        prop = report_dict["invalid_format_percentage"]
        score = max(0, weights["file_format_check"] * (1 - prop / 100))
        detailed_scores["file_format_check"] = round(score, 2)
        total_score += score

    # 6. Annotation Presence (presence of an annotation related file for machine learning tasks)
    if "annotation_presence" in report_dict:
        if report_dict["annotation_presence"] == True:
            score = weights["annotation_presence"]
        else:
            score = 0
        detailed_scores["annotation_presence"] = round(score, 2)
        total_score += score

    # 7. Timestamps Presence (presence of a timestamp related attribute in metadata)
    if "timestamps_presence" in report_dict:
        if report_dict["timestamps_presence"] == True:
            score = weights["timestamps_presence"]
        else:
            score = 0
        detailed_scores["timestamps_presence"] = round(score, 2)
        total_score += score

    # 8. Documentation Presence (Boolean)
    if "documentation_found" in report_dict:
        score = weights["documentation_presence"] if report_dict["documentation_found"] else 0
        detailed_scores["documentation_presence"] = round(score, 2)
        total_score += score
    
    # Output the results as a dictionary
    final_report = {
        "total_weights": sum(weights.values()),
        "total_score": round(total_score, 2),
        "total_percentage": round(total_score / sum(weights.values()) * 100, 2),
        "detailed_scores": detailed_scores
    }
    
    return final_report

