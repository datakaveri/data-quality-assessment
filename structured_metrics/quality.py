def check_column_missing(df, threshold=0.3):
    """
    Check which columns have missing values above a certain threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to check for missing values.
    threshold : float, optional
        Minimum proportion of missing values in a column to report. Defaults to 0.3.

    Returns
    -------
    dict
        A dictionary with three keys: "column_missing", "column_missing_count", and
        "column_missing_percentage".
        The first key maps to a dictionary where each key is a column name and the
        value is the proportion of missing values in that column, rounded to two
        decimal places.
        The second key maps to the number of columns with missing values above the
        threshold.
        The third key maps to the percentage of columns with missing values above
        the threshold relative to the total number of columns in the DataFrame.
    """

    # Number of columns (guard against zero-column DataFrame)
    num_cols = df.shape[1] if df is not None else 0

    # Build missing_report: include columns where proportion missing > threshold
    # or columns that are entirely null (but ignore truly empty Series)
    if df.empty:
        # All columns are effectively missing
        missing_report = {col: 100.0 for col in df.columns}
        return {
            "column_missing": missing_report,
            "column_missing_count": len(missing_report),
            "column_missing_percentage": 100.0,
            "number_of_columns": df.shape[1]
        }
    missing_report = {
        col : round(df[col].isnull().mean() * 100, 2)
        for col in df.columns
        if df[col].isnull().mean() > threshold or (df[col].isnull().all() and not df[col].empty)
    }
    
    # If there are no columns, return zeros (avoid division by zero)
    if num_cols == 0:
        return {
            "column_missing": {},
            "column_missing_count": 0,
            "column_missing_percentage": 0.0,
            "number_of_columns": 0,
        }

    if not missing_report:
        return {"column_missing": {},
                "column_missing_count": 0,
                "column_missing_percentage": 0.0,
                "number_of_columns": num_cols}
    else:
        return {"column_missing": missing_report,
                "column_missing_count": len(missing_report),
                "column_missing_percentage": round(len(missing_report) / num_cols * 100, 1),
                "number_of_columns": num_cols}
    
def check_row_missing(df, threshold=0.5):
    """
    Check which rows have missing values above a certain threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to check for missing values.
    threshold : float, optional
        Minimum proportion of missing values in a row to report. Defaults to 0.5.

    Returns
    -------
    dict
        A dictionary with two keys: "row_missing_count" and "row_missing_percentage".
        The first key maps to the number of rows with missing values above the threshold,
        and the second maps to the percentage of such rows relative to the total number
        of rows in the DataFrame.
    """

    count = df[df.isnull().mean(axis=1) >= threshold].shape[0]
    num_rows = df.shape[0]
    percentage = round(count / num_rows * 100, 1) if count > 0 else 0.0
    return {"row_missing_count": count,
            "row_missing_percentage": percentage,
            "number_of_rows": num_rows}

def check_row_duplicates(df):
    """
    Check which rows are exact duplicates of each other.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to check for duplicate rows.

    Returns
    -------
    dict
        A dictionary with two keys: "exact_row_duplicates" and
        "exact_row_duplicates_percentage". The first key maps to the number of
        rows with exact duplicates, and the second maps to the percentage of
        rows with exact duplicates.
    """
    # count = int(df.duplicated(keep='first').sum())
    count = int(df.duplicated().sum())
    percentage = round(df.duplicated().mean() * 100, 1) if count > 0 else 0.0
    return {"exact_row_duplicates_count": count,
            "exact_row_duplicates_percentage": percentage}  

