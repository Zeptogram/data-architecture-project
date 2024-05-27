"""
TODO docstring

Cavaleri Matteo - 875050
Gargiulo Elio - 869184
Piacente Cristian - 866020
"""

import luigi
import logging

import pandas as pd

from keras.models import Sequential, load_model
from keras.layers import Dense

import pickle

import joblib

import os

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

from utils.features_utils import drop_features, introduce_missing_values, introduce_outliers, introduce_oodv
from utils.more_rows_utils import add_rows


# Set up logger
logging.basicConfig(filename='luigi.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger('experiments-pipeline')

# Get configuration file
config = luigi.configuration.get_config()

# Dict that contains the default paths configurated in luigi.cfg
default_paths = {
    'train_csv': config.get('ExperimentFolder', 'train_csv'),
    'nn_model_file': config.get('ExperimentFolder', 'nn_model_file'),
    'svm_model_file': config.get('ExperimentFolder', 'svm_model_file'),
    'dtc_model_file': config.get('ExperimentFolder', 'dtc_model_file'),
    'drop_features_csv_name': config.get('DropFeatures', 'drop_features_csv_name'),
    'missing_values_csv_name': config.get('MissingValues', 'missing_values_csv_name'),
    'outliers_csv_name': config.get('AddOutliers', 'outliers_csv_name'),
    'oodv_csv_name': config.get('AddOODValues', 'oodv_csv_name'),
    'add_rows_random_csv_name': config.get('AddRowsRandom', 'add_rows_random_csv_name'),
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



class ExperimentFolder(luigi.Task):
    """
    TODO docstring
    """

    experiment_name = luigi.Parameter() # Mandatory
    train_csv = luigi.Parameter(default=default_paths['train_csv'])
    nn_model_file = luigi.Parameter(default=default_paths['nn_model_file'])
    svm_model_file = luigi.Parameter(default=default_paths['svm_model_file'])
    dtc_model_file = luigi.Parameter(default=default_paths['dtc_model_file'])


    def requires(self):
        # winetype_pca_train.csv, nn_model.h5, svm_model.pkl, dtc_model.pkl are needed, use a fake task
        class FakeTask(luigi.Task):
            def output(_):
                return {'train_csv': luigi.LocalTarget(self.train_csv),
                        'nn_model_file': luigi.LocalTarget(self.nn_model_file),
                        'svm_model_file': luigi.LocalTarget(self.svm_model_file),
                        'dtc_model_file': luigi.LocalTarget(self.dtc_model_file)}
            
        return FakeTask()
    

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
    # CSV
    train_csv = luigi.Parameter(default=default_paths['train_csv'])
    missing_values_csv_name = luigi.Parameter(default=default_paths['missing_values_csv_name'])


    def requires(self):
        # Dependency from drop features
        return DropFeatures(experiment_name=self.experiment_name, features_to_drop=self.features_to_drop, train_csv=self.train_csv)

    
    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Retrieve the new DataFrame, with missing values, given features
        df = introduce_missing_values(self.input().path, self.features_to_dirty_mv, self.missing_values_percentage)

        logger.info(f'Added {self.missing_values_percentage * 100}% missing values to the DataFrame, specifically on {self.features_to_dirty_mv} columns')

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
    missing_values_percentage = luigi.FloatParameter(default=0.0) 
    # For the current task
    features_to_dirty_outliers = luigi.ListParameter(default=()) 
    outliers_percentage = luigi.FloatParameter(default=0.0) 
    range_type = luigi.Parameter(default="std")
    # CSV
    train_csv = luigi.Parameter(default=default_paths['train_csv'])
    missing_values_csv_name = luigi.Parameter(default=default_paths['missing_values_csv_name'])
    outliers_csv_name = luigi.Parameter(default=default_paths['outliers_csv_name'])


    def requires(self):
        # Dependency from missing values
        return MissingValues(experiment_name=self.experiment_name, 
                            features_to_dirty_mv=self.features_to_dirty_mv,
                            missing_values_percentage=self.missing_values_percentage,
                            missing_values_csv_name=self.missing_values_csv_name, 
                            train_csv=self.train_csv, 
                            features_to_drop=self.features_to_drop)

    
    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Retrieve the new DataFrame, with outliers, given the features
        df = introduce_outliers(self.input().path, self.features_to_dirty_outliers, self.outliers_percentage, self.range_type)

        logger.info(f'Added {self.outliers_percentage * 100}% outliers to the DataFrame, specifically on {self.features_to_dirty_outliers} columns')

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
    missing_values_percentage = luigi.FloatParameter(default=0.0) 
    outliers_percentage = luigi.FloatParameter(default=0.0)
    range_type = luigi.Parameter(default="std")
    # For the current task
    features_to_dirty_oodv = luigi.ListParameter(default=()) 
    oodv_percentage = luigi.FloatParameter(default=0.0)
    # CSV
    train_csv = luigi.Parameter(default=default_paths['train_csv'])
    missing_values_csv_name = luigi.Parameter(default=default_paths['missing_values_csv_name'])
    outliers_csv_name = luigi.Parameter(default=default_paths['outliers_csv_name'])
    oodv_csv_name = luigi.Parameter(default=default_paths['oodv_csv_name'])




    def requires(self):
        # Dependency from missing values
        return AddOutliers(experiment_name=self.experiment_name, 
                            features_to_drop=self.features_to_drop,
                            features_to_dirty_mv=self.features_to_dirty_mv,
                            features_to_dirty_outliers=self.features_to_dirty_outliers,
                            missing_values_percentage=self.missing_values_percentage,
                            outliers_percentage=self.outliers_percentage,
                            range_type=self.range_type,
                            outliers_csv_name=self.outliers_csv_name,
                            missing_values_csv_name=self.missing_values_csv_name, 
                            train_csv=self.train_csv
                            )

    
    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Retrieve the new DataFrame, with out of domain values, given the features
        df = introduce_oodv(self.input().path, self.features_to_dirty_oodv, self.oodv_percentage)

        logger.info(f'Added {self.oodv_percentage * 100}% out of domain values to the DataFrame, specifically on {self.features_to_dirty_oodv} columns')

        # Save the new data in the experiment folder
        df.to_csv(self.output().path, index=False)

        logger.info('New DataFrame, with out of domain values added, saved successfully!')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(get_full_rel_path(self.experiment_name, self.oodv_csv_name))
    


class AddRowsRandom(luigi.Task):
    """
    TODO docstring
    """

    experiment_name = luigi.Parameter() # Mandatory
    features_to_drop = luigi.ListParameter(default=()) # remember to get the previous parameters
    add_rows_random_percentage = luigi.FloatParameter(default=0.0) # by default do nothing
    train_csv = luigi.Parameter(default=default_paths['train_csv'])
    add_rows_random_csv_name = luigi.Parameter(default=default_paths['add_rows_random_csv_name'])


    def requires(self):
        # TODO use Matteo's task (duplicate rows with opposite label) as the dependency.
        # Since I don't have it now, I'll use DropFeatures instead.
        return DropFeatures(experiment_name=self.experiment_name, features_to_drop=self.features_to_drop, train_csv=self.train_csv)
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Retrieve the new DataFrame, with the added rows
        df = add_rows(self.input().path, self.add_rows_random_percentage) # no min-max is passed, so the generation will be unrestricted (likely very high values)

        logger.info(f'Added {self.add_rows_random_percentage * 100}% of rows to the DataFrame, with completely random features and random target')

        # Save the new data in the experiment folder
        df.to_csv(self.output().path, index=False)

        logger.info('New DataFrame, with the added rows, saved successfully!')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(get_full_rel_path(self.experiment_name, self.add_rows_random_csv_name))