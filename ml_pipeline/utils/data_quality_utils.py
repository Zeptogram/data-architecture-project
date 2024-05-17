'''
    DATA_QUALITY_UTILS.PY

    Python file that helps the main file (data_quality.py) with the implementation
    of data quality checks, containing useful functions and separation of concerns.

    REMINDER FOR ROWS: When searching in the dataset, refer to the number + 2

Cavaleri Matteo - 875050
Gargiulo Elio - 869184
Piacente Cristian - 866020
'''

import numpy as np

# Sums missing values for each feature
def completeness_verify(df):
    missing_values = df.isnull().sum()
    return missing_values

# Calculates the ratio % of missing values for each feature
def completeness_ratio(df):
    missing_values = df.isnull().mean() * 100
    return missing_values

# Checks, with given ranges, if the features are within those ranges. If not, count the inconsistent values.
def consistency_verify(df, ranges):
    out_of_range = {}
    for feature, (min_val, max_val) in ranges.items():
        if min_val is not None and max_val is not None:
            out_of_range[feature] = len(df[(df[feature] < min_val) | (df[feature] > max_val)])
        elif min_val is not None:
            out_of_range[feature] = len(df[df[feature] < min_val])
        elif max_val is not None:
            out_of_range[feature] = len(df[df[feature] > max_val])
        else:
            out_of_range[feature] = 0
    return out_of_range

# Calculates the outliers looking at the data out of bounds (STD or IQR)
def consistency_outliers(df, ranges):
    outliers = {}
    for feature in df.columns:
        if np.issubdtype(df[feature].dtype, np.number):
            min_val, max_val = ranges.get(feature, (None, None))  # Get range for the feature
            if min_val is not None and max_val is not None:  # Check if range is specified
                feature_outliers = df[(df[feature] < min_val) | (df[feature] > max_val)]
                outliers[feature] = {
                    'count': len(feature_outliers),         
                    'rows': feature_outliers.index.tolist()  # Always +2 when looking in the file
                }
    return outliers

# Calculates the bounds for each feature, to have a more accurate, yet not perfect idea 
# of our data consistency. Mean and Threshold * Standard deviation has been used. 
def consistency_calculate_bounds_std(data, feature_ranges, threshold = 5):
    updated_feature_ranges = {}
    for feature, (lower_bound, upper_bound) in feature_ranges.items():
        # Uses mean and standard deviation for each feature
        mean = data[feature].mean()
        std = data[feature].std()
        # Calculate upper bound using mean + std multiplied for a threshold (EX. 5)
        upper_bound_new = mean + threshold * std
        # Same for lower bound, but bounds the bottom limit to 0, for non-negative numbers
        lower_bound_new = max(mean - threshold * std, 0)
        # Update the ranges
        updated_feature_ranges[feature] = (lower_bound_new, upper_bound_new)
    return updated_feature_ranges

# Calculates the bounds using Interquartile Range (IQR)
def consistency_calculate_bounds_iqr(data, feature_ranges, threshold = 4):
    updated_feature_ranges = {}
    for feature in feature_ranges:
        q1 = data[feature].quantile(0.25)
        q3 = data[feature].quantile(0.75)
        IQR = q3 - q1
        # Calculate upper and lower bounds based on IQR and threshold
        upper_bound_new = q3 + threshold * IQR
        lower_bound_new = max(q1 - threshold * IQR, 0)  # Bound the bottom limit to 0 for non-negative numbers
        # Update the ranges
        updated_feature_ranges[feature] = (lower_bound_new, upper_bound_new)
    return updated_feature_ranges

# Verifies how unique each feature is
def uniqueness_verify_unique(df):
    unique_counts = df.nunique()
    return unique_counts

# Verifies if there are any duplicates, if so counts the duplicated data + sum
def uniqueness_verify_duplicates(df):
    duplicate_indexes = df[df.duplicated()].index.tolist()
    duplicate_sum = df.duplicated().sum()
    return duplicate_indexes, duplicate_sum

# Calculates the ratio % of unique values for each feature
def uniqueness_ratio(df):
    unique_values = df.nunique() / len(df) * 100
    return unique_values

# Verifies, given the expected types, that each feature is considered as the expected type.
def accuracy_verify(df, expected_types):
    accuracy_results = {}
    for feature, expected_type in expected_types.items():
        actual_type = df[feature].dtype
        accuracy_results[feature] = actual_type == expected_type
    return accuracy_results


'''
    END OF FILE

'''