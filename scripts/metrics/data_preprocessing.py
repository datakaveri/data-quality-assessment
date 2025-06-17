import pandas as pd

def preProcess(df, input1, input2):
    """
    Preprocesses the dataframe by converting observation time to datetime,
    sorting by uniqueID and time, and calculating inter-arrival times (IAT).
    
    Args:
        df: DataFrame containing observationDateTime and uniqueID columns
        input1: Column name for unique identifier
        input2: Column name for datetime
        
    Returns:
        Processed DataFrame with IAT column.
    """
    # Make a copy to avoid SettingWithCopyWarning
    df_processed = df.copy()
    
    # Convert datetime column  
    df_processed[input2] = pd.to_datetime(df_processed[input2])
    
    # Keep only required columns
    df_processed = df_processed[[input2, input1]]
    
    # Sort by uniqueID first, then by datetime
    df_processed = df_processed.sort_values(by=[input1, input2], ascending=[True, True])
    
    # Calculate IAT using simple diff (matches original logic)
    df_processed['IAT'] = df_processed[input2].diff().dt.total_seconds()
    
    # Apply the original filtering logic but correctly
    # Keep only rows where IAT >= 0 (this also removes NaN values from first row)
    df_processed = df_processed[df_processed['IAT'] >= 0]
    
    # Reset index
    df_processed = df_processed.reset_index(drop=True)
    
    return df_processed
import pandas as pd

def preProcess(df, input1, input2):
    """
    Preprocesses the dataframe by converting observation time to datetime,
    sorting by uniqueID and time, and calculating inter-arrival times (IAT).
    
    Args:
        df: DataFrame containing observationDateTime and uniqueID columns
        input1: Column name for unique identifier
        input2: Column name for datetime
        
    Returns:
        Processed DataFrame with IAT column
    """
    # Make a copy to avoid SettingWithCopyWarning
    df_processed = df.copy()
    
    # Convert datetime column  
    df_processed[input2] = pd.to_datetime(df_processed[input2])
    
    # Keep only required columns
    df_processed = df_processed[[input2, input1]]
    
    # Sort by uniqueID first, then by datetime
    df_processed = df_processed.sort_values(by=[input1, input2], ascending=[True, True])
    
    # Calculate IAT using simple diff (matches original logic)
    df_processed['IAT'] = df_processed[input2].diff().dt.total_seconds()
    
    # Apply the original filtering logic but correctly
    # Keep only rows where IAT >= 0 (this also removes NaN values from first row)
    df_processed = df_processed[df_processed['IAT'] >= 0]
    
    # Reset index
    df_processed = df_processed.reset_index(drop=True)
    
    return df_processed