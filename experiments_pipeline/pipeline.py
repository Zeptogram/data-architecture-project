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

import ultraimport

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

from sklearn.utils import shuffle

from utils.features_utils import drop_features, introduce_missing_values, introduce_outliers, introduce_oodv, get_ranges
from utils.more_rows_utils import add_rows

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
    outliers_csv_name = luigi.Parameter(default=default_paths['outliers_csv_name'])


    def requires(self):
        # Dependency from missing values
        return MissingValues(experiment_name=self.experiment_name, 
                             features_to_drop=self.features_to_drop,
                             features_to_dirty_mv=self.features_to_dirty_mv,
                             missing_values_percentage=self.missing_values_percentage,
                             train_csv=self.train_csv)

    
    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Original training set DataFrame, needed for outliers
        original_train_df = pd.read_csv(self.train_csv)

        # Retrieve the new DataFrame, with outliers, given the features
        df = introduce_outliers(self.input().path, original_train_df, self.features_to_dirty_outliers, self.outliers_percentage, self.range_type)

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
    oodv_csv_name = luigi.Parameter(default=default_paths['oodv_csv_name'])


    def requires(self):
        # Dependency from add outliers
        return AddOutliers(experiment_name=self.experiment_name, 
                           features_to_drop=self.features_to_drop,
                           features_to_dirty_mv=self.features_to_dirty_mv,
                           missing_values_percentage=self.missing_values_percentage,
                           features_to_dirty_outliers=self.features_to_dirty_outliers,
                           outliers_percentage=self.outliers_percentage,
                           range_type=self.range_type,
                           train_csv=self.train_csv)

    
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
    features_to_drop = luigi.ListParameter(default=())
    features_to_dirty_mv = luigi.ListParameter(default=()) 
    features_to_dirty_outliers = luigi.ListParameter(default=()) 
    missing_values_percentage = luigi.FloatParameter(default=0.0) 
    outliers_percentage = luigi.FloatParameter(default=0.0)
    range_type = luigi.Parameter(default="std")
    features_to_dirty_oodv = luigi.ListParameter(default=()) 
    oodv_percentage = luigi.FloatParameter(default=0.0)
    add_rows_random_percentage = luigi.FloatParameter(default=0.0) # by default do nothing
    train_csv = luigi.Parameter(default=default_paths['train_csv'])
    add_rows_random_csv_name = luigi.Parameter(default=default_paths['add_rows_random_csv_name'])


    def requires(self):
        # TODO use Matteo's task (duplicate rows with opposite label) as the dependency.
        # Since I don't have it now, I'll use AddOODValues instead. I'll take care of the dependencies (with all the parameters etc.) when we merge everything.
        return AddOODValues(experiment_name=self.experiment_name, 
                            features_to_drop=self.features_to_drop, 
                            features_to_dirty_mv=self.features_to_dirty_mv,
                            features_to_dirty_outliers=self.features_to_dirty_outliers,
                            missing_values_percentage=self.missing_values_percentage,
                            outliers_percentage=self.outliers_percentage,
                            range_type=self.range_type,
                            features_to_dirty_oodv=self.features_to_dirty_oodv,
                            oodv_percentage=self.oodv_percentage,
                            train_csv=self.train_csv)
    

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
    add_rows_random_percentage = luigi.FloatParameter(default=0.0)
    add_rows_domain_percentage = luigi.FloatParameter(default=0.0)
    train_csv = luigi.Parameter(default=default_paths['train_csv'])
    add_rows_domain_csv_name = luigi.Parameter(default=default_paths['add_rows_domain_csv_name'])


    def requires(self):
        # Dependency from add rows random
        return AddRowsRandom(experiment_name=self.experiment_name,
                             features_to_drop=self.features_to_drop, 
                             features_to_dirty_mv=self.features_to_dirty_mv,
                             features_to_dirty_outliers=self.features_to_dirty_outliers,
                             missing_values_percentage=self.missing_values_percentage,
                             outliers_percentage=self.outliers_percentage,
                             range_type=self.range_type,
                             features_to_dirty_oodv=self.features_to_dirty_oodv,
                             oodv_percentage=self.oodv_percentage,
                             add_rows_random_percentage=self.add_rows_random_percentage, 
                             train_csv=self.train_csv)
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Original training set DataFrame, needed for get_ranges
        original_train_df = pd.read_csv(self.train_csv)

        # Get the domain ranges using Mean +- 10 * Std
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
    add_rows_random_percentage = luigi.FloatParameter(default=0.0)
    add_rows_domain_percentage = luigi.FloatParameter(default=0.0)
    train_csv = luigi.Parameter(default=default_paths['train_csv'])
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
                                                 add_rows_random_percentage=self.add_rows_random_percentage, 
                                                 add_rows_domain_percentage=self.add_rows_domain_percentage, 
                                                 train_csv=self.train_csv),
                'initial_files': FakeTask(train_csv=self.train_csv)}
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Read the final dirty csv
        final_dirty_train_df = pd.read_csv(self.input()['final_dirty_csv'].path)

        # Split into X_train and y_train
        X_train = final_dirty_train_df.drop('type', axis=1)
        y_train = final_dirty_train_df['type']

        logger.info('Retrieved the final dirty training set')

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

        # Create the whole set DataFrame (needed for CV) by appending the test set to the training set and shuffling
        df = pd.concat([final_dirty_train_df, test_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

        # Split into X and y
        X = df.drop('type', axis=1).to_numpy()
        y = df['type']

        logger.info('Generated the whole shuffled set')

        # Define the neural network
        nn_model_naive = Sequential()

        # A network with a number of initial neurons that is equal to the number of kept PCA components
        n_features = len(X_train.columns)
        nn_model_naive.add(Dense(n_features, input_shape=(n_features,), activation='relu'))
        # An output neuron with a sigmoid activation function (boolean target)
        nn_model_naive.add(Dense(1, activation='sigmoid'))

        # Compile the model
        nn_model_naive.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        logger.info('Created the Neural Network')

        # Fit the Neural Network on the dirty training set
        nn_model_naive.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

        logger.info('Trained the Neural Network on the dirty training set!')

        # Define the SVM
        svm_model_naive = svm.SVC(kernel='linear', C=0.001, random_state=42)

        logger.info('Created the SVM')

        # Fit the SVM on the dirty training set
        svm_model_naive.fit(X_train, y_train)

        logger.info('Trained the SVM on the dirty training set!')

        # Define the Decision Tree
        dtc_model_naive = DecisionTreeClassifier(random_state=42)

        logger.info('Created the Decision Tree')

        # Fit the Decision Tree on the dirty training set
        dtc_model_naive.fit(X_train, y_train)

        logger.info('Trained the Decision Tree on the dirty training set!')

        # Map the model names to their instances
        models_dict = {
            'Neural Network': nn_model_naive,
            'SVM': svm_model_naive,
            'Decision Tree': dtc_model_naive
        }

        # Dictionary structure which will be converted to DataFrame
        # Keys = column names
        # Values = column data, one for each row
        metrics_dict = {
            'experiment_name': [self.experiment_name] * 3,
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

        # Append the data to metrics.csv
        with open(self.output().path, 'a') as f:
            # The header gets written only if the csv is empty
            metrics_df.to_csv(f, mode='a', header=f.tell()==0, index=False, lineterminator='\n')

        logger.info('Appended to csv the performance evaluation for each model fitted on the dirty training set!')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(get_full_rel_path(self.experiment_name, self.metrics_csv_name))