import numpy as np

def calculate_error_statistics(error_values: np.array, start_time=None, end_time=None) -> dict:
    """
    Calculate standard statistics for error values.
    
    Args:
        error_values: Numpy array of error values
        start_time: Optional start time of the analysis
        end_time: Optional end time of the analysis
        
    Returns:
        dict: Dictionary containing calculated statistics
    """
    stats = {
        "minimum": np.min(error_values),
        "maximum": np.max(error_values),
        "median": np.median(error_values),
        "std_dev": np.std(error_values),
        "mean": np.mean(error_values),
        "sample_count": len(error_values),
        "rms": np.sqrt(np.mean(np.square(error_values))),
        "start_time_since_recording": start_time,
        "end_time_since_recording": end_time,
    }
    
    return stats

def print_stats(stats: dict, title: str, decimal_places: int = 4) -> None:
    """
    Print statistics.

    Args:
        stats: Dictionary of statistics
        title: Title for the statistics block
        decimal_places: Number of decimal places (default: 4)
    """
    print(f"\n{title}:")
    for key, value in stats.items():
        if isinstance(value, (int, bool)):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.{decimal_places}f}")

def align_time_series(timestamps1, values1, timestamps2, values2):
    """
    Aligns two time series by interpolating to a common time base.
    
    Args:
        timestamps1: First series timestamps
        values1: First series values
        timestamps2: Second series timestamps
        values2: Second series values
        
    Returns:
        tuple: (common_timestamps, interpolated_values1, interpolated_values2)
    """
    # Find the overlapping time range
    start_time = max(timestamps1[0], timestamps2[0])
    end_time = min(timestamps1[-1], timestamps2[-1])
    
    # Create a common time base with the highest sampling rate
    dt1 = np.mean(np.diff(timestamps1))
    dt2 = np.mean(np.diff(timestamps2))
    dt = min(dt1, dt2)  # Use the smaller time step
    
    common_timestamps = np.arange(start_time, end_time, dt)
    
    # Interpolate both signals to the common timebase
    interpolated_values1 = np.interp(common_timestamps, timestamps1, values1)
    interpolated_values2 = np.interp(common_timestamps, timestamps2, values2)
    
    return common_timestamps, interpolated_values1, interpolated_values2