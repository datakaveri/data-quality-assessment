def duplicatesMetric(df, input1, input2):
    """
    Computes duplicate detection metric based on specified columns.
    Must be called before inter-arrival time creation.
    
    Args:
        df: DataFrame to check for duplicates
        input1: First column name to check for duplicates
        input2: Second column name to check for duplicates
        
    Returns:
        Duplicates metric score (0-1, where 1 means no duplicates).
    """
    totalDataPackets = len(df)
    
    # Handle empty DataFrame case
    if totalDataPackets == 0:
        return 1.0  # Perfect score for empty DataFrame (no duplicates possible)
    
    uniqueCount = len(df.drop_duplicates(subset=[input1, input2]))
    dupeCount = totalDataPackets - uniqueCount
    duplicatesMetricScore = 1 - dupeCount / totalDataPackets
    return round(duplicatesMetricScore, 3)