"""
TODO docstring

Cavaleri Matteo - 875050
Gargiulo Elio - 869184
Piacente Cristian - 866020
"""

import luigi
import logging

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

import os

import ultraimport

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

from utils.features_utils import drop_features, introduce_missing_values, introduce_outliers, introduce_oodv, get_ranges
from utils.more_rows_utils import add_rows, duplicate_rows
from utils.label_utils import flip_labels

from hyperimpute.plugins.imputers import Imputers

get_global_metrics = ultraimport(f"{os.getcwd()}/../ml_pipeline/utils/evaluation.py", "get_global_metrics")
get_confidence_intervals = ultraimport(f"{os.getcwd()}/../ml_pipeline/utils/evaluation.py", "get_confidence_intervals")


# Set up logger
logging.basicConfig(filename='luigi.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger('experiments-pipeline')

# Get configuration file
config = luigi.configuration.get_config()

# Dict that contains the default parameters configurated in luigi.cfg
default_paths = {
    'train_csv': config.get('ExperimentFolder', 'train_csv'),
    'test_csv': config.get('ExperimentFolder', 'test_csv'),
    'drop_features_csv_name': config.get('DropFeatures', 'drop_features_csv_name'),
    'missing_values_csv_name': config.get('MissingValues', 'missing_values_csv_name'),
    'outliers_csv_name': config.get('AddOutliers', 'outliers_csv_name'),
    'oodv_csv_name': config.get('AddOODValues', 'oodv_csv_name'),
    'flip_labels_csv_name': config.get('FlipLabels','flip_labels_csv_name'),
    'duplicate_rows_same_label_csv_name': config.get('DuplicateRowsSameLabel', 'duplicate_rows_same_label_csv_name'),
    'duplicate_rows_opposite_label_csv_name': config.get('DuplicateRowsOppositeLabel', 'duplicate_rows_opposite_label_csv_name'),
    'add_rows_random_csv_name': config.get('AddRowsRandom', 'add_rows_random_csv_name'),
    'add_rows_domain_csv_name': config.get('AddRowsDomain', 'add_rows_domain_csv_name'),
    'metrics_csv_name': config.get('FitPerformanceEval', 'metrics_csv_name'),
}

# Experiment folder
experiment_folder = ''

# Init global variable if not already done
def init_global_var(experiment_name):
    global experiment_folder
    if experiment_folder == '':
        experiment_folder = f'experiments/{experiment_name}'

# Retrieve a relative path w.r.t the esperiment name
# suffix is the substring of the path after the experiment folder
def get_full_rel_path(experiment_name, suffix):
    init_global_var(experiment_name) # Initialize the global variable experiment_folder
    return f'{experiment_folder}/{suffix}'


class DirectoryTarget(luigi.Target):
    """
    TODO docstring
    """
    
    def __init__(self, path):
        self.path = path

    # Luigi's complete method override
    def complete(self):
        return os.path.isdir(self.path)

    # Luigi's complete method override
    def exists(self):
        return os.path.isdir(self.path)



class FakeTask(luigi.Task):
    train_csv = luigi.Parameter(default=default_paths['train_csv'])
    test_csv = luigi.Parameter(default=default_paths['test_csv'])

    def output(self):
        return {'train_csv': luigi.LocalTarget(self.train_csv),
                'test_csv': luigi.LocalTarget(self.test_csv)}



class ExperimentFolder(luigi.Task):
    """
    TODO docstring
    """

    experiment_name = luigi.Parameter() # Mandatory
    train_csv = luigi.Parameter(default=default_paths['train_csv'])
    test_csv = luigi.Parameter(default=default_paths['test_csv'])

    def requires(self):
        # winetype_pca_train.csv, winetype_pca_test.csv are needed
        return FakeTask(train_csv=self.train_csv,
                        test_csv=self.test_csv)
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Create the experiment folder
        os.makedirs(self.output().path)

        logger.info('Experiment folder created successfully!')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return DirectoryTarget(get_full_rel_path(self.experiment_name, ''))



class DropFeatures(luigi.Task):
    """
    TODO docstring
    """

    experiment_name = luigi.Parameter() # Mandatory
    features_to_drop = luigi.ListParameter(default=()) # using an empty tuple by default since Luigi creates a tuple instead of a list, by default do nothing
    train_csv = luigi.Parameter(default=default_paths['train_csv'])
    drop_features_csv_name = luigi.Parameter(default=default_paths['drop_features_csv_name'])


    def requires(self):
        # the experiment folder is needed
        return ExperimentFolder(experiment_name=self.experiment_name, train_csv=self.train_csv)
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Retrieve the new DataFrame, without the given features
        df = drop_features(self.train_csv, self.features_to_drop)

        logger.info(f'Dropped the features {self.features_to_drop}')

        # Save the new data in the experiment folder
        df.to_csv(self.output().path, index=False)

        logger.info('New DataFrame, without the given features, saved successfully!')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(get_full_rel_path(self.experiment_name, self.drop_features_csv_name))



class MissingValues(luigi.Task):
    """
    TODO docstring
    """

    experiment_name = luigi.Parameter() # Mandatory
    # Param for Dependency
    features_to_drop = luigi.ListParameter(default=())
    # For the current task
    features_to_dirty_mv = luigi.ListParameter(default=()) 
    missing_values_percentage = luigi.FloatParameter(default=0.0)
    wine_types_to_consider_missing_values = luigi.ListParameter(default=('red', 'white'))
    # CSV
    train_csv = luigi.Parameter(default=default_paths['train_csv'])
    missing_values_csv_name = luigi.Parameter(default=default_paths['missing_values_csv_name'])


    def requires(self):
        # Dependency from drop features
        return DropFeatures(experiment_name=self.experiment_name, features_to_drop=self.features_to_drop, train_csv=self.train_csv)

    
    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Retrieve the new DataFrame, with missing values, given the features
        df = introduce_missing_values(self.input().path, self.wine_types_to_consider_missing_values, self.features_to_dirty_mv, self.missing_values_percentage)

        logger.info(f'Added {self.missing_values_percentage * 100}% missing values to the DataFrame, specifically on {self.features_to_dirty_mv} columns with target/s {self.wine_types_to_consider_missing_values}')

        # Save the new data in the experiment folder
        df.to_csv(self.output().path, index=False)

        logger.info('New DataFrame, with missing values added, saved successfully!')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(get_full_rel_path(self.experiment_name, self.missing_values_csv_name))

    

class AddOutliers(luigi.Task):
    """
    TODO docstring
    """

    experiment_name = luigi.Parameter() # Mandatory
    # Param for Dependency
    features_to_drop = luigi.ListParameter(default=()) 
    features_to_dirty_mv = luigi.ListParameter(default=())
    wine_types_to_consider_missing_values = luigi.ListParameter(default=('red', 'white'))
    missing_values_percentage = luigi.FloatParameter(default=0.0) 
    # For the current task
    features_to_dirty_outliers = luigi.ListParameter(default=()) 
    outliers_percentage = luigi.FloatParameter(default=0.0) 
    wine_types_to_consider_outliers = luigi.ListParameter(default=('red', 'white'))
    range_type = luigi.Parameter(default="std")
    # CSV
    train_csv = luigi.Parameter(default=default_paths['train_csv'])
    outliers_csv_name = luigi.Parameter(default=default_paths['outliers_csv_name'])


    def requires(self):
        # Dependency from missing values
        return MissingValues(experiment_name=self.experiment_name, 
                             features_to_drop=self.features_to_drop,
                             features_to_dirty_mv=self.features_to_dirty_mv,
                             missing_values_percentage=self.missing_values_percentage,
                             wine_types_to_consider_missing_values=self.wine_types_to_consider_missing_values,
                             train_csv=self.train_csv)

    
    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Original training set DataFrame, needed for outliers
        original_train_df = pd.read_csv(self.train_csv)

        # Retrieve the new DataFrame, with outliers, given the features
        df = introduce_outliers(self.input().path, original_train_df, self.wine_types_to_consider_outliers, self.features_to_dirty_outliers, self.outliers_percentage, self.range_type)

        logger.info(f'Added {self.outliers_percentage * 100}% outliers to the DataFrame, specifically on {self.features_to_dirty_outliers} columns with target/s {self.wine_types_to_consider_outliers}')

        # Save the new data in the experiment folder
        df.to_csv(self.output().path, index=False)

        logger.info('New DataFrame, with outliers added, saved successfully!')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(get_full_rel_path(self.experiment_name, self.outliers_csv_name))



class AddOODValues(luigi.Task):
    """
    TODO docstring
    """

    experiment_name = luigi.Parameter() # Mandatory
    # Param for Dependency
    features_to_drop = luigi.ListParameter(default=()) 
    features_to_dirty_mv = luigi.ListParameter(default=()) 
    features_to_dirty_outliers = luigi.ListParameter(default=()) 
    wine_types_to_consider_missing_values = luigi.ListParameter(default=('red', 'white'))
    wine_types_to_consider_outliers = luigi.ListParameter(default=('red', 'white'))
    missing_values_percentage = luigi.FloatParameter(default=0.0) 
    outliers_percentage = luigi.FloatParameter(default=0.0)
    range_type = luigi.Parameter(default="std")
    # For the current task
    features_to_dirty_oodv = luigi.ListParameter(default=()) 
    oodv_percentage = luigi.FloatParameter(default=0.0)
    wine_types_to_consider_oodv = luigi.ListParameter(default=('red', 'white'))
    # CSV
    train_csv = luigi.Parameter(default=default_paths['train_csv'])
    oodv_csv_name = luigi.Parameter(default=default_paths['oodv_csv_name'])


    def requires(self):
        # Dependency from add outliers
        return AddOutliers(experiment_name=self.experiment_name, 
                           features_to_drop=self.features_to_drop,
                           features_to_dirty_mv=self.features_to_dirty_mv,
                           missing_values_percentage=self.missing_values_percentage,
                           features_to_dirty_outliers=self.features_to_dirty_outliers,
                           wine_types_to_consider_missing_values=self.wine_types_to_consider_missing_values,
                           wine_types_to_consider_outliers=self.wine_types_to_consider_missing_values,
                           outliers_percentage=self.outliers_percentage,
                           range_type=self.range_type,
                           train_csv=self.train_csv)

    
    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Retrieve the new DataFrame, with out of domain values, given the features
        df = introduce_oodv(self.input().path, self.wine_types_to_consider_oodv, self.features_to_dirty_oodv, self.oodv_percentage)

        logger.info(f'Added {self.oodv_percentage * 100}% out of domain values to the DataFrame, specifically on {self.features_to_dirty_oodv} columns with target/s {self.wine_types_to_consider_oodv} ')

        # Save the new data in the experiment folder
        df.to_csv(self.output().path, index=False)

        logger.info('New DataFrame, with out of domain values added, saved successfully!')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(get_full_rel_path(self.experiment_name, self.oodv_csv_name))
    


class FlipLabels(luigi.Task):
    """
    TODO docstring
    """

    experiment_name = luigi.Parameter() # Mandatory
    # Dependencies
    features_to_drop = luigi.ListParameter(default=()) 
    features_to_dirty_mv = luigi.ListParameter(default=()) 
    features_to_dirty_outliers = luigi.ListParameter(default=()) 
    missing_values_percentage = luigi.FloatParameter(default=0.0) 
    outliers_percentage = luigi.FloatParameter(default=0.0)
    range_type = luigi.Parameter(default="std")
    features_to_dirty_oodv = luigi.ListParameter(default=()) 
    oodv_percentage = luigi.FloatParameter(default=0.0)
    train_csv = luigi.Parameter(default=default_paths['train_csv'])
    wine_types_to_consider_missing_values = luigi.ListParameter(default=('red', 'white'))
    wine_types_to_consider_outliers = luigi.ListParameter(default=('red', 'white'))
    wine_types_to_consider_oodv = luigi.ListParameter(default=('red', 'white'))
    # Current Task
    flip_percentage_red = luigi.FloatParameter(default=0.0)
    flip_percentage_white = luigi.FloatParameter(default=0.0)
    flip_labels_csv_name = luigi.Parameter(default=default_paths['flip_labels_csv_name'])


    def requires(self):
        # Dependency from add out of domain values
        return AddOODValues(experiment_name=self.experiment_name,
                            features_to_drop=self.features_to_drop,
                            features_to_dirty_mv=self.features_to_dirty_mv,
                            features_to_dirty_outliers=self.features_to_dirty_outliers,
                            missing_values_percentage=self.missing_values_percentage,
                            wine_types_to_consider_missing_values=self.wine_types_to_consider_missing_values,
                            wine_types_to_consider_outliers=self.wine_types_to_consider_missing_values,
                            wine_types_to_consider_oodv=self.wine_types_to_consider_oodv,
                            outliers_percentage=self.outliers_percentage,
                            range_type=self.range_type,
                            features_to_dirty_oodv=self.features_to_dirty_oodv,
                            oodv_percentage=self.oodv_percentage,
                            train_csv=self.train_csv)
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Retrieve the new DataFrame, with flipped labels given the red and white percentages
        df = flip_labels(self.input().path, self.flip_percentage_red, self.flip_percentage_white)

        logger.info(f'{self.flip_percentage_red * 100}% of the red wines and {self.flip_percentage_white * 100}% of the white wines have been flipped')

        # Save the new data in the experiment folder
        df.to_csv(self.output().path, index=False)

        logger.info('New DataFrame, with flipped labels, saved successfully!')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(get_full_rel_path(self.experiment_name, self.flip_labels_csv_name))



class DuplicateRowsSameLabel(luigi.Task):
    """
    TODO docstring
    """

    experiment_name = luigi.Parameter() # Mandatory
    # Dependencies
    features_to_drop = luigi.ListParameter(default=()) 
    features_to_dirty_mv = luigi.ListParameter(default=()) 
    features_to_dirty_outliers = luigi.ListParameter(default=()) 
    missing_values_percentage = luigi.FloatParameter(default=0.0) 
    outliers_percentage = luigi.FloatParameter(default=0.0)
    range_type = luigi.Parameter(default="std")
    features_to_dirty_oodv = luigi.ListParameter(default=()) 
    oodv_percentage = luigi.FloatParameter(default=0.0)
    train_csv = luigi.Parameter(default=default_paths['train_csv'])
    flip_percentage_red = luigi.FloatParameter(default=0.0)
    flip_percentage_white = luigi.FloatParameter(default=0.0)
    wine_types_to_consider_missing_values = luigi.ListParameter(default=('red', 'white'))
    wine_types_to_consider_outliers = luigi.ListParameter(default=('red', 'white'))
    wine_types_to_consider_oodv = luigi.ListParameter(default=('red', 'white'))
    # Current Task
    wine_types_to_consider_same_label = luigi.ListParameter(default=('red', 'white'))
    duplicate_rows_same_label_percentage = luigi.FloatParameter(default=0.0)
    duplicate_rows_same_label_csv_name = luigi.Parameter(default=default_paths['duplicate_rows_same_label_csv_name'])
    

    def requires(self):
        # Dependency from flip labels
        return FlipLabels(experiment_name=self.experiment_name,
                          features_to_drop=self.features_to_drop,
                          features_to_dirty_mv=self.features_to_dirty_mv,
                          features_to_dirty_outliers=self.features_to_dirty_outliers,
                          missing_values_percentage=self.missing_values_percentage,
                          wine_types_to_consider_missing_values=self.wine_types_to_consider_missing_values,
                          wine_types_to_consider_outliers=self.wine_types_to_consider_missing_values,
                          wine_types_to_consider_oodv=self.wine_types_to_consider_oodv,
                          outliers_percentage=self.outliers_percentage,
                          range_type=self.range_type,
                          features_to_dirty_oodv=self.features_to_dirty_oodv,
                          oodv_percentage=self.oodv_percentage,
                          train_csv=self.train_csv,
                          flip_percentage_red=self.flip_percentage_red,
                          flip_percentage_white=self.flip_percentage_white)
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Retrieve the new DataFrame, with duplicate rows (with same label) given the wine types to consider and the percentage
        df = duplicate_rows(self.input().path, self.wine_types_to_consider_same_label, self.duplicate_rows_same_label_percentage, flip_label=False)

        # Save the new data in the experiment folder
        df.to_csv(self.output().path, index=False)

        logger.info('New DataFrame, with duplicate rows with same label, saved successfully!')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(get_full_rel_path(self.experiment_name, self.duplicate_rows_same_label_csv_name)) 



class DuplicateRowsOppositeLabel(luigi.Task):
    """
    TODO docstring
    """

    experiment_name = luigi.Parameter() # Mandatory
    # Dependencies
    features_to_drop = luigi.ListParameter(default=()) 
    features_to_dirty_mv = luigi.ListParameter(default=()) 
    features_to_dirty_outliers = luigi.ListParameter(default=()) 
    missing_values_percentage = luigi.FloatParameter(default=0.0) 
    outliers_percentage = luigi.FloatParameter(default=0.0)
    range_type = luigi.Parameter(default="std")
    features_to_dirty_oodv = luigi.ListParameter(default=()) 
    oodv_percentage = luigi.FloatParameter(default=0.0)
    train_csv = luigi.Parameter(default=default_paths['train_csv'])
    flip_percentage_red = luigi.FloatParameter(default=0.0)
    flip_percentage_white = luigi.FloatParameter(default=0.0)
    wine_types_to_consider_same_label = luigi.ListParameter(default=('red', 'white'))
    wine_types_to_consider_missing_values = luigi.ListParameter(default=('red', 'white'))
    wine_types_to_consider_outliers = luigi.ListParameter(default=('red', 'white'))
    wine_types_to_consider_oodv = luigi.ListParameter(default=('red', 'white'))
    duplicate_rows_same_label_percentage = luigi.FloatParameter(default=0.0)
    # Current Task
    wine_types_to_consider_opposite_label = luigi.ListParameter(default=('red', 'white'))
    duplicate_rows_opposite_label_percentage = luigi.FloatParameter(default=0.0)
    duplicate_rows_opposite_label_csv_name = luigi.Parameter(default=default_paths['duplicate_rows_opposite_label_csv_name'])
    

    def requires(self):
        # Dependency from duplicate rows same label
        return DuplicateRowsSameLabel(experiment_name=self.experiment_name,
                                      features_to_drop=self.features_to_drop,
                                      features_to_dirty_mv=self.features_to_dirty_mv,
                                      features_to_dirty_outliers=self.features_to_dirty_outliers,
                                      missing_values_percentage=self.missing_values_percentage,
                                      outliers_percentage=self.outliers_percentage,
                                      wine_types_to_consider_missing_values=self.wine_types_to_consider_missing_values,
                                      wine_types_to_consider_outliers=self.wine_types_to_consider_missing_values,
                                      wine_types_to_consider_oodv=self.wine_types_to_consider_oodv,
                                      range_type=self.range_type,
                                      features_to_dirty_oodv=self.features_to_dirty_oodv,
                                      oodv_percentage=self.oodv_percentage,
                                      train_csv=self.train_csv,
                                      flip_percentage_red=self.flip_percentage_red,
                                      flip_percentage_white=self.flip_percentage_white,
                                      wine_types_to_consider_same_label=self.wine_types_to_consider_same_label,
                                      duplicate_rows_same_label_percentage=self.duplicate_rows_same_label_percentage)
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Retrieve the new DataFrame, with duplicate rows (with opposite label) given the wine types to consider and the percentage, flip_label=True by default
        df = duplicate_rows(self.input().path, self.wine_types_to_consider_opposite_label, self.duplicate_rows_opposite_label_percentage)

        # Save the new data in the experiment folder
        df.to_csv(self.output().path, index=False)

        logger.info('New DataFrame, with duplicate rows with opposite label, saved successfully!')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(get_full_rel_path(self.experiment_name, self.duplicate_rows_opposite_label_csv_name)) 



class AddRowsRandom(luigi.Task):
    """
    TODO docstring
    """

    experiment_name = luigi.Parameter() # Mandatory
    features_to_drop = luigi.ListParameter(default=())
    features_to_dirty_mv = luigi.ListParameter(default=()) 
    features_to_dirty_outliers = luigi.ListParameter(default=()) 
    missing_values_percentage = luigi.FloatParameter(default=0.0) 
    outliers_percentage = luigi.FloatParameter(default=0.0)
    range_type = luigi.Parameter(default="std")
    features_to_dirty_oodv = luigi.ListParameter(default=()) 
    oodv_percentage = luigi.FloatParameter(default=0.0)
    train_csv = luigi.Parameter(default=default_paths['train_csv'])
    flip_percentage_red = luigi.FloatParameter(default=0.0)
    flip_percentage_white = luigi.FloatParameter(default=0.0)
    wine_types_to_consider_same_label = luigi.ListParameter(default=('red', 'white'))
    duplicate_rows_same_label_percentage = luigi.FloatParameter(default=0.0)
    wine_types_to_consider_opposite_label = luigi.ListParameter(default=('red', 'white'))
    duplicate_rows_opposite_label_percentage = luigi.FloatParameter(default=0.0)
    wine_types_to_consider_missing_values = luigi.ListParameter(default=('red', 'white'))
    wine_types_to_consider_outliers = luigi.ListParameter(default=('red', 'white'))
    wine_types_to_consider_oodv = luigi.ListParameter(default=('red', 'white'))
    # Task specific parameters
    add_rows_random_percentage = luigi.FloatParameter(default=0.0) # by default do nothing
    add_rows_random_csv_name = luigi.Parameter(default=default_paths['add_rows_random_csv_name'])


    def requires(self):
        # Dependency from duplicate rows opposite label
        return DuplicateRowsOppositeLabel(experiment_name=self.experiment_name,
                                          features_to_drop=self.features_to_drop,
                                          features_to_dirty_mv=self.features_to_dirty_mv,
                                          features_to_dirty_outliers=self.features_to_dirty_outliers,
                                          missing_values_percentage=self.missing_values_percentage,
                                          outliers_percentage=self.outliers_percentage,
                                          wine_types_to_consider_missing_values=self.wine_types_to_consider_missing_values,
                                          wine_types_to_consider_outliers=self.wine_types_to_consider_missing_values,
                                          wine_types_to_consider_oodv=self.wine_types_to_consider_oodv,
                                          range_type=self.range_type,
                                          features_to_dirty_oodv=self.features_to_dirty_oodv,
                                          oodv_percentage=self.oodv_percentage,
                                          train_csv=self.train_csv,
                                          flip_percentage_red=self.flip_percentage_red,
                                          flip_percentage_white=self.flip_percentage_white,
                                          wine_types_to_consider_same_label=self.wine_types_to_consider_same_label,
                                          duplicate_rows_same_label_percentage=self.duplicate_rows_same_label_percentage,
                                          wine_types_to_consider_opposite_label=self.wine_types_to_consider_opposite_label,
                                          duplicate_rows_opposite_label_percentage=self.duplicate_rows_opposite_label_percentage)
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Retrieve the new DataFrame, with the added rows
        df = add_rows(self.input().path, self.add_rows_random_percentage) # no ranges are passed, so the generation will be unrestricted (probably very high values)

        logger.info(f'Added {self.add_rows_random_percentage * 100}% of rows to the DataFrame, with completely random features and random target')

        # Save the new data in the experiment folder
        df.to_csv(self.output().path, index=False)

        logger.info('New DataFrame, with the added rows, saved successfully!')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(get_full_rel_path(self.experiment_name, self.add_rows_random_csv_name))
    


class AddRowsDomain(luigi.Task):
    """
    TODO docstring
    """

    experiment_name = luigi.Parameter() # Mandatory
    features_to_drop = luigi.ListParameter(default=())
    features_to_dirty_mv = luigi.ListParameter(default=()) 
    features_to_dirty_outliers = luigi.ListParameter(default=()) 
    missing_values_percentage = luigi.FloatParameter(default=0.0) 
    outliers_percentage = luigi.FloatParameter(default=0.0)
    range_type = luigi.Parameter(default="std")
    features_to_dirty_oodv = luigi.ListParameter(default=()) 
    oodv_percentage = luigi.FloatParameter(default=0.0)
    train_csv = luigi.Parameter(default=default_paths['train_csv'])
    flip_percentage_red = luigi.FloatParameter(default=0.0)
    flip_percentage_white = luigi.FloatParameter(default=0.0)
    wine_types_to_consider_same_label = luigi.ListParameter(default=('red', 'white'))
    duplicate_rows_same_label_percentage = luigi.FloatParameter(default=0.0)
    wine_types_to_consider_opposite_label = luigi.ListParameter(default=('red', 'white'))
    duplicate_rows_opposite_label_percentage = luigi.FloatParameter(default=0.0)
    add_rows_random_percentage = luigi.FloatParameter(default=0.0)
    wine_types_to_consider_missing_values = luigi.ListParameter(default=('red', 'white'))
    wine_types_to_consider_outliers = luigi.ListParameter(default=('red', 'white'))
    wine_types_to_consider_oodv = luigi.ListParameter(default=('red', 'white'))
    # Task specific parameters
    add_rows_domain_percentage = luigi.FloatParameter(default=0.0)
    add_rows_domain_csv_name = luigi.Parameter(default=default_paths['add_rows_domain_csv_name'])


    def requires(self):
        # Dependency from add rows random
        return AddRowsRandom(experiment_name=self.experiment_name,
                             features_to_drop=self.features_to_drop,
                             features_to_dirty_mv=self.features_to_dirty_mv,
                             features_to_dirty_outliers=self.features_to_dirty_outliers,
                             missing_values_percentage=self.missing_values_percentage,
                             outliers_percentage=self.outliers_percentage,
                             wine_types_to_consider_missing_values=self.wine_types_to_consider_missing_values,
                             wine_types_to_consider_outliers=self.wine_types_to_consider_missing_values,
                             wine_types_to_consider_oodv=self.wine_types_to_consider_oodv,
                             range_type=self.range_type,
                             features_to_dirty_oodv=self.features_to_dirty_oodv,
                             oodv_percentage=self.oodv_percentage,
                             train_csv=self.train_csv,
                             flip_percentage_red=self.flip_percentage_red,
                             flip_percentage_white=self.flip_percentage_white,
                             wine_types_to_consider_same_label=self.wine_types_to_consider_same_label,
                             duplicate_rows_same_label_percentage=self.duplicate_rows_same_label_percentage,
                             wine_types_to_consider_opposite_label=self.wine_types_to_consider_opposite_label,
                             duplicate_rows_opposite_label_percentage=self.duplicate_rows_opposite_label_percentage,
                             add_rows_random_percentage=self.add_rows_random_percentage)
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Original training set DataFrame, needed for get_ranges
        original_train_df = pd.read_csv(self.train_csv)

        # Get the domain ranges using Mean +- 3 * Std
        ranges_std = get_ranges(original_train_df, original_train_df.columns[1:], threshold_std = 3)

        # Retrieve the new DataFrame, with the added rows
        df = add_rows(self.input().path, self.add_rows_domain_percentage, ranges = ranges_std)

        logger.info(f'Added {self.add_rows_random_percentage * 100}% of rows to the DataFrame, with features in the domain ranges and random target')

        # Save the new data in the experiment folder
        df.to_csv(self.output().path, index=False)

        logger.info('New DataFrame, with the added rows, saved successfully!')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(get_full_rel_path(self.experiment_name, self.add_rows_domain_csv_name))
    


class FitPerformanceEval(luigi.Task):
    """
    TODO docstring
    """

    experiment_name = luigi.Parameter() # Mandatory
    features_to_drop = luigi.ListParameter(default=())
    features_to_dirty_mv = luigi.ListParameter(default=()) 
    features_to_dirty_outliers = luigi.ListParameter(default=()) 
    missing_values_percentage = luigi.FloatParameter(default=0.0) 
    outliers_percentage = luigi.FloatParameter(default=0.0)
    range_type = luigi.Parameter(default="std")
    features_to_dirty_oodv = luigi.ListParameter(default=()) 
    oodv_percentage = luigi.FloatParameter(default=0.0)
    train_csv = luigi.Parameter(default=default_paths['train_csv'])
    test_csv = luigi.Parameter(default=default_paths['test_csv'])
    flip_percentage_red = luigi.FloatParameter(default=0.0)
    flip_percentage_white = luigi.FloatParameter(default=0.0)
    wine_types_to_consider_same_label = luigi.ListParameter(default=('red', 'white'))
    duplicate_rows_same_label_percentage = luigi.FloatParameter(default=0.0)
    wine_types_to_consider_opposite_label = luigi.ListParameter(default=('red', 'white'))
    duplicate_rows_opposite_label_percentage = luigi.FloatParameter(default=0.0)
    add_rows_random_percentage = luigi.FloatParameter(default=0.0)
    add_rows_domain_percentage = luigi.FloatParameter(default=0.0)
    wine_types_to_consider_missing_values = luigi.ListParameter(default=('red', 'white'))
    wine_types_to_consider_outliers = luigi.ListParameter(default=('red', 'white'))
    wine_types_to_consider_oodv = luigi.ListParameter(default=('red', 'white'))
    # Task specific parameters
    metrics_csv_name = luigi.Parameter(default=default_paths['metrics_csv_name'])



    def requires(self):
        # Dependency from add rows domain
        return {'final_dirty_csv': AddRowsDomain(experiment_name=self.experiment_name,
                                                 features_to_drop=self.features_to_drop,
                                                 features_to_dirty_mv=self.features_to_dirty_mv,
                                                 features_to_dirty_outliers=self.features_to_dirty_outliers,
                                                 missing_values_percentage=self.missing_values_percentage,
                                                 outliers_percentage=self.outliers_percentage,
                                                 range_type=self.range_type,
                                                 features_to_dirty_oodv=self.features_to_dirty_oodv,
                                                 oodv_percentage=self.oodv_percentage,
                                                 wine_types_to_consider_missing_values=self.wine_types_to_consider_missing_values,
                                                 wine_types_to_consider_outliers=self.wine_types_to_consider_missing_values,
                                                 wine_types_to_consider_oodv=self.wine_types_to_consider_oodv,
                                                 train_csv=self.train_csv,
                                                 flip_percentage_red=self.flip_percentage_red,
                                                 flip_percentage_white=self.flip_percentage_white,
                                                 wine_types_to_consider_same_label=self.wine_types_to_consider_same_label,
                                                 duplicate_rows_same_label_percentage=self.duplicate_rows_same_label_percentage,
                                                 wine_types_to_consider_opposite_label=self.wine_types_to_consider_opposite_label,
                                                 duplicate_rows_opposite_label_percentage=self.duplicate_rows_opposite_label_percentage,
                                                 add_rows_random_percentage=self.add_rows_random_percentage, 
                                                 add_rows_domain_percentage=self.add_rows_domain_percentage),
                'initial_files': FakeTask(train_csv=self.train_csv, test_csv=self.test_csv)}
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Read the final dirty csv
        final_dirty_train_df = pd.read_csv(self.input()['final_dirty_csv'].path)

        # Copy of the final dirty training set with float values instead of bool for the target, to be used with the Imputers
        float_dirty_train_df = final_dirty_train_df.copy()
        float_dirty_train_df['type'] = float_dirty_train_df['type'].astype(float)

        # Split into X_train and y_train
        X_train = final_dirty_train_df.drop('type', axis=1)
        y_train = final_dirty_train_df['type']

        logger.info('Retrieved the final dirty training set')

        # Neural Network by default interprets missing values as 0.0 (which leads to unexpected behavior): exploit and compare two imputation strategies (mean and EM)
        # SVM errors while fitting if there are missing values, so the following imputed DataFrames will be used by both NN and SVM
        mean_dirty_train_df = Imputers().get("mean").fit_transform(float_dirty_train_df.copy())
        mean_dirty_train_df.columns = final_dirty_train_df.columns
        mean_dirty_train_df['type'] = mean_dirty_train_df['type'].astype(bool)
        em_dirty_train_df = Imputers().get("EM").fit_transform(float_dirty_train_df.copy())
        em_dirty_train_df.columns = final_dirty_train_df.columns
        em_dirty_train_df['type'] = em_dirty_train_df['type'].astype(bool)

        # Split into X_train_mean and X_train_em (y_train is the same as before, since there can't be missing values)
        X_train_mean = mean_dirty_train_df.drop('type', axis=1)
        X_train_em = em_dirty_train_df.drop('type', axis=1)

        logger.info('Retrieved two imputed training sets using mean and EM algorithm')

        # Read winetype_pca_test.csv
        test_df = pd.read_csv(self.input()['initial_files']['test_csv'].path)

        # Remove from the test set the features which aren't in the dirty training set
        for feature in test_df.columns[1:]:
            if not (feature in final_dirty_train_df):
                test_df = test_df.drop(feature, axis=1)

        # Split into X_test and y_test
        X_test = test_df.drop('type', axis=1)
        y_test = test_df['type']

        logger.info('Retrieved the test set without eventual dropped features')

        # Create the whole sets' DataFrame (needed for CV) by appending the test set to the training set and shuffling
        df = pd.concat([final_dirty_train_df, test_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        df_mean = pd.concat([mean_dirty_train_df, test_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        df_em = pd.concat([em_dirty_train_df, test_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split into X, X_mean, X_em and y
        X = df.drop('type', axis=1).to_numpy()
        X_mean = df_mean.drop('type', axis=1).to_numpy()
        X_em = df_em.drop('type', axis=1).to_numpy()
        y = df['type']

        logger.info('Generated the whole shuffled sets')

        # Define the neural networks, one for mean and one for EM
        nn_model_naive_mean = Sequential()
        nn_model_naive_em = Sequential()

        # The networks have a number of initial neurons that is equal to the number of kept PCA components
        # The output neurons use a sigmoid activation function (boolean target)
        n_features = len(X_train.columns)
        nn_model_naive_mean.add(Dense(n_features, input_shape=(n_features,), activation='relu'))
        nn_model_naive_mean.add(Dense(1, activation='sigmoid'))
        nn_model_naive_em.add(Dense(n_features, input_shape=(n_features,), activation='relu'))
        nn_model_naive_em.add(Dense(1, activation='sigmoid'))

        # Compile the models
        nn_model_naive_mean.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        nn_model_naive_em.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        logger.info('Created two Neural Network instances')

        # Fit the Neural Networks
        nn_model_naive_mean.fit(X_train_mean, y_train, epochs=10, batch_size=32, validation_split=0.2)
        nn_model_naive_em.fit(X_train_em, y_train, epochs=10, batch_size=32, validation_split=0.2)

        logger.info('Trained the Neural Networks on imputed training sets!')

        # We need two SVM instances
        svm_model_naive_mean = svm.SVC(kernel='linear', C=0.001, random_state=42)
        svm_model_naive_em = svm.SVC(kernel='linear', C=0.001, random_state=42)

        logger.info('Created two SVM instances')

        # Fit the SVMs
        svm_model_naive_mean.fit(X_train_mean, y_train)
        svm_model_naive_em.fit(X_train_em, y_train)

        logger.info('Trained the SVMs on imputed training sets!')

        # Define the Decision Tree
        dtc_model_naive = DecisionTreeClassifier(random_state=42)

        logger.info('Created the Decision Tree')

        # Fit the Decision Tree on the dirty training set
        dtc_model_naive.fit(X_train, y_train)

        logger.info('Trained the Decision Tree on the dirty training set!')

        # Map the model names to their instances
        models_dict = {
            'Neural Network (mean)': nn_model_naive_mean,
            'Neural Network (EM)': nn_model_naive_em,
            'SVM (mean)': svm_model_naive_mean,
            'SVM (EM)': svm_model_naive_em,
            'Decision Tree': dtc_model_naive
        }

        # Dictionary structure which will be converted to DataFrame
        # Keys = column names
        # Values = column data, one for each row
        metrics_dict = {
            'experiment_name': [self.experiment_name] * len(models_dict),
            'model_name': list(models_dict.keys()),

            'accuracy': [],
            'accuracy_interval_lower': [],
            'accuracy_interval_upper': [],

            'precision': [],
            'precision_interval_lower': [],
            'precision_interval_upper': [],

            'recall': [],
            'recall_interval_lower': [],
            'recall_interval_upper': [],

            'f1_score': [],
            'f1_score_interval_lower': [],
            'f1_score_interval_upper': []
        }

        # For each model fill the structure with the metrics data
        for model_name in metrics_dict['model_name']:

            # Model instance
            model = models_dict[model_name]

            # Global metrics
            global_metrics = get_global_metrics(model, X_test, y_test)

            # Add the global metrics to the structure
            metrics_dict['accuracy'].append(global_metrics['accuracy'])
            metrics_dict['precision'].append(global_metrics['precision'])
            metrics_dict['recall'].append(global_metrics['recall'])
            metrics_dict['f1_score'].append(global_metrics['f1_score'])

            logger.info(f'Got the global metrics for {model_name}')

            # 95% confidence intervals
            if '(mean)' in model_name:
                confidence_intervals = get_confidence_intervals(model, X_mean, y)
            elif '(EM)' in model_name:
                confidence_intervals = get_confidence_intervals(model, X_em, y)
            else:
                confidence_intervals = get_confidence_intervals(model, X, y)

            # Add the 95% confidence intervals to the structure
            metrics_dict['accuracy_interval_lower'].append(confidence_intervals['accuracy_interval'][0])
            metrics_dict['accuracy_interval_upper'].append(confidence_intervals['accuracy_interval'][1])
            metrics_dict['precision_interval_lower'].append(confidence_intervals['precision_interval'][0])
            metrics_dict['precision_interval_upper'].append(confidence_intervals['precision_interval'][1])
            metrics_dict['recall_interval_lower'].append(confidence_intervals['recall_interval'][0])
            metrics_dict['recall_interval_upper'].append(confidence_intervals['recall_interval'][1])
            metrics_dict['f1_score_interval_lower'].append(confidence_intervals['f1_score_interval'][0])
            metrics_dict['f1_score_interval_upper'].append(confidence_intervals['f1_score_interval'][1])
            
            logger.info(f'Got the 95% confidence intervals for {model_name}')
        
        # Convert the dictionary structure to DataFrame
        metrics_df = pd.DataFrame(metrics_dict)

        # Save the data to metrics.csv
        metrics_df.to_csv(self.output().path, index=False)

        logger.info('Saved the performance evaluation for each model fitted on the dirty training set on csv!')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(get_full_rel_path(self.experiment_name, self.metrics_csv_name))