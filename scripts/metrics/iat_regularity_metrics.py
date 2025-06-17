import scipy.stats as sps
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def computeModeDeviation(dataframe):
    """
    Computes mode deviation for a given dataframe series.
   
    Args:
<<<<<<< HEAD
        dataframe: Series or array of values.
=======
        dataframe: Series or array of values
>>>>>>> 9c5f2989031ba54019bec835b7ecb3f5768f2dcf
       
    Returns:
        Mode deviation value
    """
    mode_result = sps.mode(dataframe)
    
    # Handle different scipy versions
    if hasattr(mode_result, 'mode'):
        # Newer scipy versions (>= 1.9.0)
        modeValue = mode_result.mode
        if hasattr(modeValue, '__len__') and len(modeValue) > 0:
            modeValue = modeValue[0]
    else:
        # Older scipy versions
        try:
            modeValue = mode_result[0][0]
        except (IndexError, TypeError):
            modeValue = mode_result[0] if hasattr(mode_result[0], '__len__') else mode_result[0]
    
    deviations = abs(dataframe - modeValue)
    modeDeviation = deviations.sum() / len(dataframe)
    return modeDeviation

def iatRegularityMetric(dataframe):
    """
    Computes IAT regularity metric using Relative Absolute Error (RAE).
   
    Args:
        dataframe: DataFrame with 'IAT' column
       
    Returns:
        IAT regularity metric score (0-1)
    """
    modeValue = dataframe['IAT'].mode()[0]
    goodCount = 0
    badCount = 0
    count = 0
    for iat in dataframe['IAT']:
        RAE_i = (np.abs(iat - modeValue)) / modeValue
        if RAE_i <= 0.5:
            goodCount += 1 - 2 * RAE_i
            count += 1
        else:
            badCount += 2 * RAE_i
        totalCount = count + badCount
    iatRegularityMetricScore = goodCount / totalCount
    return round(iatRegularityMetricScore, 3)