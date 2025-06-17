import scipy.stats as sps
import numpy as np

def iatOutliersMetric(dataframe):
    """
    Computes IAT outliers metric using modified Z-score approach.
    Uses median absolute deviation (MAD) for robust outlier detection.
   
    Args:
        dataframe: DataFrame with 'IAT' column
       
    Returns:
        IAT outliers metric score (0-1, where 1 means no outliers)
    """
    df = dataframe
    data = df['IAT'].dropna()
    
    # Handle empty data after dropping NaN
    if len(data) == 0:
        raise ValueError("No valid IAT data found after removing NaN values")
    
    # adaptive threshold using regular z-score
    mode = sps.mode(data)[0]
    mad = np.median(np.abs(data - mode))
    print(mode)
    threshold_mod_z_score = 3.5  
    # defining fences
    # Handle division by zero when mad is 0
    if mad == 0:
        outliers = []  # No outliers when all values are identical
    else:
        outliers = [x for x in df['IAT'] if not np.isnan(x) and ((0.6745 * (x - mode)) / mad) > threshold_mod_z_score]
    outlierNumber = len(outliers)
    totalDataPackets = len(df)
    iatMetricOutlierScore = 1 - (outlierNumber / totalDataPackets)
    return round(iatMetricOutlierScore, 3)