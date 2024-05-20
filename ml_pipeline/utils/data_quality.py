"""
Module for implementing data quality checks on the wine dataset.

This module leverages utility functions from utils.data_quality_utils 
to perform data quality assessments across four dimensions: 
completeness, consistency, uniqueness, and accuracy.

Cavaleri Matteo - 875050
Gargiulo Elio - 869184
Piacente Cristian - 866020
"""

import utils.data_quality_utils as utils


#    DEFAULT VALUES

#    Ranges, Types and Thresholds that will be used in the following functions if
#    not specified by the caller or edited here

# Feature Ranges of the expected values. Any value that it outside the expected range is considered inconsistent
feature_ranges = {
    'fixed acidity': (0, None),  # Check for non-negative values
    'volatile acidity': (0, None),  # Check for non-negative values
    'citric acid': (0, None),  # Check for non-negative values
    'residual sugar': (0, None),  # Check for non-negative values
    'chlorides': (0, None),  # Check for non-negative values
    'free sulfur dioxide': (0, None),  # Check for non-negative values
    'total sulfur dioxide': (0, None),  # Check for non-negative values
    'density': (0, None),  # Check for non-negative values 
    'pH': (0, 14),  # Default Range for pH
    'sulphates': (0, None),  # Check for non-negative values
    'alcohol': (0, 100),  # % Alcohol goes necessarely from 0 to 100%
}



# Feature: type that has to be respected
expected_types = {
        'fixed acidity': 'float64',
        'volatile acidity': 'float64',
        'citric acid': 'float64',
        'residual sugar': 'float64',
        'chlorides': 'float64',
        'free sulfur dioxide': 'float64',
        'total sulfur dioxide': 'float64',
        'density': 'float64',
        'pH': 'float64',
        'sulphates': 'float64',
        'alcohol': 'float64',
        'type': 'bool'  
    }

# Values for thresholds, used to find outliers
threshold_std = 5
threshold_iqr = 4



#    COMPLETENESS

#    Quality Measure that ensures that the dataset that will be used does not contain 
#    any missing values for every feature recorded in the set.

def completeness_test(dataset):
    # Checks if there are any missing values
    missing_values = utils.completeness_verify(dataset)
    # Gets the ratio for each feature of missing values
    completeness_ratio = utils.completeness_ratio(dataset)
    # Return the missing_values and ratio
    return missing_values, completeness_ratio
    


#    CONSISTENCY

#    Quality measure that ensures that the dataset used maintains internal coherence 
#    and adheres to predefined rules or constraints. This involves checking for any 
#    inconsistencies or contradictions within the dataset itself.

def consistency_test(dataset, ranges = feature_ranges):
    # Implement consistency checks based on domain knowledge or specific rules
    inconsistent_values_default = utils.consistency_verify(dataset, ranges)
    # Finds more accurate ranges for another check
    std_bounds = utils.consistency_calculate_bounds_std(dataset, ranges, threshold_std)
    iqr_bounds = utils.consistency_calculate_bounds_iqr(dataset, ranges, threshold_iqr)
    # Checks with calculated bounds
    inconsistent_values_bounded_std= utils.consistency_verify(dataset, std_bounds)
    inconsistent_values_bounded_iqr= utils.consistency_verify(dataset, iqr_bounds)
    # Finds the outliers for each feature with ranges obtaned by mean and std) and iqr
    outliers_std = utils.consistency_outliers(dataset, std_bounds)
    outliers_iqr = utils.consistency_outliers(dataset, iqr_bounds)
    # Return the inconsistent values and outliers
    return inconsistent_values_default, inconsistent_values_bounded_std, inconsistent_values_bounded_iqr, outliers_std, outliers_iqr, std_bounds, iqr_bounds



#    UNIQUENESS

#    Quality measure that guarantees the dataset's elements are distinct and singular, 
#    devoid of any redundant or repeated entries. It involves confirming that each 
#    data point within the dataset stands out on its own, contributing uniquely to 
#    the dataset's richness and integrity.

def uniqueness_test(dataset):
    # Counts unique values for each feature
    unique_counts = utils.uniqueness_verify_unique(dataset)
    # Finds any duplicate data with their row and sum
    duplicate_counts, duplicate_sum = utils.uniqueness_verify_duplicates(dataset)
    # Gets the ratio of unique values
    unique_ratio = utils.uniqueness_ratio(dataset)
    return unique_counts, duplicate_counts, duplicate_sum, unique_ratio



#    ACCURACY

#    Quality measure that verifies if the data is actually correctly
#    used and considered. In this case, it's important that the types
#    assigned to each feature reflects its meaning in the real world

def accuracy_test(dataset, types = expected_types):
    # Verifies the correct type for each feature
    accuracy_results = utils.accuracy_verify(dataset, types)
    return accuracy_results