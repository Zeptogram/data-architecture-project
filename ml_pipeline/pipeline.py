"""
Pipeline for data preprocessing, transforming, modeling, and evaluating three wine dataset models.
It also checks four data quality measures, w.r.t a threshold.
This script uses the Luigi library to manage tasks in a sequential and dependent way.

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

from utils.evaluation import get_global_metrics, get_confidence_intervals

import utils.data_quality as dq

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

import tensorflow as tf
from tensorflow import keras


# Set up logger
logging.basicConfig(filename='luigi.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger('ml-pipeline')

# Get configuration file
config = luigi.configuration.get_config()

# Dict that contains the default paths configurated in luigi.cfg
default_paths = {
    'input_csv': config.get('DataPreprocessing', 'input_csv'),
    'cleaned_csv': config.get('DataPreprocessing', 'cleaned_csv'),
    'transformed_csv': config.get('DataTransformation', 'transformed_csv'),
    'pca_csv': config.get('PCATask', 'pca_csv'),
    'train_csv': config.get('SplitDataset', 'train_csv'),
    'test_csv': config.get('SplitDataset', 'test_csv'),
    'nn_model_file': config.get('NNModel', 'nn_model_file'),
    'nn_history_file': config.get('NNModel', 'nn_history_file'),
    'svm_model_file': config.get('SVMModel', 'svm_model_file'),
    'dtc_model_file': config.get('DTCModel', 'dtc_model_file'),
    'metrics_csv': config.get('PerformanceEval', 'metrics_csv')
}

# Threshold for data quality measures
dq_threshold = 3 # 3%
dq_count_threshold = int(5295 / 100 * dq_threshold) # 158


class InMemoryTarget(luigi.Target):
    """
    A custom target which is based on a flag, in the memory, instead of a file.
    This adds support for tasks which don't have a file as the output.
    """

    def __init__(self):
        self.completed = False

    # Luigi's complete method override
    def complete(self):
        return self.completed
    
    # Luigi's exists method override
    def exists(self):
        return self.completed



class DataPreprocessing(luigi.Task):
    """
    Task to preprocess the data by removing missing values and duplicates.
    
    Parameters:
    - input-csv: Path to the input csv file containing raw wine data. Default: 'datasets/winetype.csv'
    - cleaned-csv: Path to the output csv file for preprocessed data. Default: 'datasets/winetype_cleaned.csv'
    """

    input_csv = luigi.Parameter(default=default_paths['input_csv'])
    cleaned_csv = luigi.Parameter(default=default_paths['cleaned_csv'])


    def requires(self):
        # winetype.csv is needed, use a fake task
        class FakeTask(luigi.Task):
            def output(_):
                return luigi.LocalTarget(self.input_csv)
            
        return FakeTask()
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Read winetype.csv
        df = pd.read_csv(self.input().path)

        logger.info('Retrieved the original dataset')

        logger.info(f'Dataset dimension before preprocessing: {df.shape}')

        logger.info(f'Missing values:\n{df.isnull().sum()}')

        # Drop rows with missing values
        df.dropna(inplace=True)

        logger.info('Dropped rows with missing values')

        logger.info(f'Duplicated rows: {df.duplicated().sum()}')

        # Drop duplicated rows
        df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=False)

        logger.info('Dropped duplicated rows')

        logger.info(f'Dataset dimension after preprocessing: {df.shape}')

        # Save the preprocessed data to winetype_cleaned.csv
        df.to_csv(self.output().path, index=False)

        logger.info('Preprocessed data saved successfully!')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(self.cleaned_csv)
    


class DataTransformation(luigi.Task):
    """
    Task to transform the data by encoding categorical variables and dropping unnecessary columns.
    
    Parameters:
    - input-csv: Path to the input csv file containing raw wine data. Default: 'datasets/winetype.csv'
    - transformed-csv: Path for the output dataset with transformed features. Default: 'datasets/winetype_transformed.csv'
    """

    input_csv = luigi.Parameter(default=default_paths['input_csv'])
    transformed_csv = luigi.Parameter(default=default_paths['transformed_csv'])


    def requires(self):
        # winetype_cleaned.csv is needed
        return DataPreprocessing(input_csv=self.input_csv)
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Read winetype_cleaned.csv
        df = pd.read_csv(self.input().path)

        logger.info('Retrieved the preprocessed dataset')

        logger.info(f'Data types before encoding and casting:\n{df.dtypes}')

        # Label Encoding (red = False, white = True)
        df['type'] = LabelEncoder().fit_transform(df['type']) 
        df['type'] = df['type'].astype(bool) # Cast to bool

        logger.info('Label encoded the target, which is now bool (red = False, white = True)')

        # Cast the feature quality to categorical, since its values can be between 0 and 10
        df['quality'] = df['quality'].astype('category')

        logger.info('Casted the feature quality to categorical')

        logger.info(f'Data types after encoding and casting:\n{df.dtypes}')

        logger.info(f'Number of features before dropping the feature quality: {df.shape[1] - 1}')

        # Drop the feature quality, please refer to the notebook to find out why
        df.drop(columns='quality', inplace=True)

        logger.info(f'Number of features after dropping the feature quality: {df.shape[1] - 1}')

        # Save the transformed data to winetype_transformed.csv
        df.to_csv(self.output().path, index=False)

        logger.info('Transformed data saved successfully!')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(self.transformed_csv)



class PCATask(luigi.Task):
    """
    Task to perform Principal Component Analysis (PCA) for dimensionality reduction.
    
    Parameters:
    - input-csv: Path to the input csv file containing raw wine data. Default: 'datasets/winetype.csv'
    - pca-csv: Path to the output csv file for the PCA-transformed data. Default: 'datasets/winetype_pca.csv'
    """

    input_csv = luigi.Parameter(default=default_paths['input_csv'])
    pca_csv = luigi.Parameter(default=default_paths['pca_csv'])


    def requires(self):
        # winetype_transformed.csv is needed
        return DataTransformation(input_csv=self.input_csv)
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Read winetype_transformed.csv
        df = pd.read_csv(self.input().path)

        logger.info('Retrieved the transformed dataset')

        # Only consider numerical features (exclude the target)
        indexes = list(range(1, 12))
        features = [df.columns[i] for i in indexes]

        logger.info(f'Numerical features: {features}')

        # Standardize the features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[features])

        logger.info('Scaled the data')

        # Dimensionality reduction with 5 components
        pca = PCA(n_components=5).fit(scaled_data)
        pca_data = pca.transform(scaled_data)

        logger.info('Applied PCA to the data, with n_components = 5')

        # Convert the PCA data to DataFrame
        pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(pca_data.shape[1])])

        # Add the target to the DataFrame
        pca_df.insert(0, 'type', df['type'])

        # Save the PCA data to winetype_pca.csv
        pca_df.to_csv(self.output().path, index=False)

        logger.info('PCA data saved successfully!')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(self.pca_csv)



class SplitDataset(luigi.Task):
    """
    Splits the PCA dataset into training and testing sets.
    Outputs two files, one for training and one for testing.
    
    Parameters:
    - input-csv: Path to the input csv file containing raw wine data. Default: 'datasets/winetype.csv'
    - train-csv: Path for the output training set (after PCA). Default: 'datasets/winetype_pca_train.csv'
    - test-csv: Path for the output testing set (after PCA). Default: 'datasets/winetype_pca_test.csv'
    """

    input_csv = luigi.Parameter(default=default_paths['input_csv'])
    train_csv = luigi.Parameter(default=default_paths['train_csv'])
    test_csv = luigi.Parameter(default=default_paths['test_csv'])
    

    def requires(self):
        # winetype_pca.csv is needed
        return PCATask(input_csv=self.input_csv)
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Read winetype_pca.csv
        df = pd.read_csv(self.input().path)

        # Split into X and y
        X = df.drop('type', axis=1)
        y = df['type']

        logger.info('Retrieved the PCA dataset')

        # 80% training, 20% test
        train_size = 0.8
        test_size = 0.2

        # Split into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        logger.info('Split into training set (80%) and test set (20%)')

        logger.info(f'Total dimension: {X.shape}')
        logger.info(f'Training set dimension: {X_train.shape}')
        logger.info(f'Test set dimension: {X_test.shape}')

        # Get the whole training set DataFrame
        train_df = pd.concat([y_train, X_train], axis=1)

        # Get the whole test set DataFrame
        test_df = pd.concat([y_test, X_test], axis=1)

        # Save the training set to winetype_pca_train.csv
        train_df.to_csv(self.output()['train_csv'].path, index=False)

        logger.info('Training set saved successfully!')

        # Save the test set to winetype_pca_test.csv
        test_df.to_csv(self.output()['test_csv'].path, index=False)
        
        logger.info('Test set saved successfully!')
        logger.info(f'Finished task {self.__class__.__name__}')
        

    def output(self):
        return {'train_csv': luigi.LocalTarget(self.train_csv),
                'test_csv': luigi.LocalTarget(self.test_csv)}
    


class NNModel(luigi.Task):
    """
    Task to train and save a Neural Network model.
    
    Parameters:
    - input-csv: Path to the input csv file containing raw wine data. Default: 'datasets/winetype.csv'
    - nn-model-file: Path to save the trained Neural Network model. Default: 'models/nn_model.h5'
    - nn-history-file: Path to save the training history. Default: 'models/nn_history.pkl'
    """

    input_csv = luigi.Parameter(default=default_paths['input_csv'])
    nn_model_file = luigi.Parameter(default=default_paths['nn_model_file'])
    nn_history_file = luigi.Parameter(default=default_paths['nn_history_file'])


    def requires(self):
        # winetype_pca_train.csv is needed
        return SplitDataset(input_csv=self.input_csv)
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Read winetype_pca_train.csv
        train_df = pd.read_csv(self.input()['train_csv'].path)

        # Split into X_train and y_train
        X_train = train_df.drop('type', axis=1)
        y_train = train_df['type']

        logger.info('Retrieved the training set')

        # Define the neural network
        nn_model_naive = Sequential()

        # A network with a number of initial neurons that is equal to the number of PCA components (5)
        nn_model_naive.add(Dense(5, input_shape=(5,), activation='relu'))
        # An output neuron with a sigmoid activation function (boolean target)
        nn_model_naive.add(Dense(1, activation='sigmoid'))

        # Compile the model
        nn_model_naive.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        logger.info('Built the model')

        # Train the model
        history_naive = nn_model_naive.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

        logger.info('Trained the model')

        # Save the entire model to a HDF5 file
        nn_model_naive.save(self.output()['nn_model_file'].path)

        logger.info('Model saved successfully!')

        # Create the history path directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output()['nn_history_file'].path), exist_ok=True)

        # Save the training history to a .pkl file
        with open(self.output()['nn_history_file'].path, 'wb') as f:
            pickle.dump(history_naive.history, f)

        logger.info('Training history saved successfully!')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return {'nn_model_file': luigi.LocalTarget(self.nn_model_file),
                'nn_history_file': luigi.LocalTarget(self.nn_history_file)}
    


class SVMModel(luigi.Task):
    """
    Task to train and save a Support Vector Machine (SVM) model.
    
    Parameters:
    - input-csv: Path to the input csv file containing raw wine data. Default: 'datasets/winetype.csv'
    - svm-model-file: Path to save the trained SVM model. Default: 'models/svm_model.pkl'
    """

    input_csv = luigi.Parameter(default=default_paths['input_csv'])
    svm_model_file = luigi.Parameter(default=default_paths['svm_model_file'])


    def requires(self):
        # winetype_pca_train.csv is needed
        return SplitDataset(input_csv=self.input_csv)
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Read winetype_pca_train.csv
        train_df = pd.read_csv(self.input()['train_csv'].path)

        # Split into X_train and y_train
        X_train = train_df.drop('type', axis=1)
        y_train = train_df['type']

        logger.info('Retrieved the training set')

        # Define the SVM
        svm_model_naive = svm.SVC(kernel='linear', random_state=42)

        logger.info('Built the model')

        # Train the model
        svm_model_naive.fit(X_train, y_train)

        logger.info('Trained the model')

        # Save the entire model to a .pkl file
        joblib.dump(svm_model_naive, self.output().path)

        logger.info('Model saved successfully!')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(self.svm_model_file)
    


class DTCModel(luigi.Task):
    """
    Task to train and save a Decision Tree Classifier model.
    
    Parameters:
    - input-csv: Path to the input csv file containing raw wine data. Default: 'datasets/winetype.csv'
    - dtc-model-file: Path to save the trained Decision Tree model. Default: 'models/dtc_model.pkl'
    """

    input_csv = luigi.Parameter(default=default_paths['input_csv'])
    dtc_model_file = luigi.Parameter(default=default_paths['dtc_model_file'])


    def requires(self):
        # winetype_pca_train.csv is needed
        return SplitDataset(input_csv=self.input_csv)
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Read winetype_pca_train.csv
        train_df = pd.read_csv(self.input()['train_csv'].path)

        # Split into X_train and y_train
        X_train = train_df.drop('type', axis=1)
        y_train = train_df['type']

        logger.info('Retrieved the training set')

        # Define the Decision Tree
        dtc_model_naive = DecisionTreeClassifier(random_state=42)

        logger.info('Built the model')

        # Train the model
        dtc_model_naive.fit(X_train, y_train)

        logger.info('Trained the model')

        # Save the entire model to a .pkl file
        joblib.dump(dtc_model_naive, self.output().path)

        logger.info('Model saved successfully!')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(self.dtc_model_file)
    


class PerformanceEval(luigi.Task):
    """
    Evaluates the trained models' performance.
    
    Outputs a csv file with performance metrics calculated both on the test set (global metrics)
    and using Stratified 10-fold Cross Validation (95% confidence intervals).
    
    Parameters:
    - input-csv: Path to the input csv file containing raw wine data. Default: 'datasets/winetype.csv'
    - metrics-csv: Path to save the performance metrics csv file. Default: 'performance/metrics.csv'
    """

    input_csv = luigi.Parameter(default=default_paths['input_csv'])
    metrics_csv = luigi.Parameter(default=default_paths['metrics_csv'])


    def requires(self):
        # winetype_pca.csv, nn_model.h5, nn_history.pkl, svm_model.pkl, dtc_model.pkl and winetype_pca_test.csv are needed
        return {'pca_csv': PCATask(input_csv=self.input_csv),
                'nn_files': NNModel(input_csv=self.input_csv),
                'svm_model_file': SVMModel(input_csv=self.input_csv),
                'dtc_model_file': DTCModel(input_csv=self.input_csv),
                'splitted_dataset_csv': SplitDataset(input_csv=self.input_csv)}


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Read winetype_pca.csv
        df = pd.read_csv(self.input()['pca_csv'].path)

        # Split into X and y
        X = df.drop('type', axis=1).to_numpy()
        y = df['type']

        logger.info('Retrieved the dataset')

        # Read winetype_pca_test.csv
        test_df = pd.read_csv(self.input()['splitted_dataset_csv']['test_csv'].path)

        # Split into X_test and y_test
        X_test = test_df.drop('type', axis=1)
        y_test = test_df['type']

        logger.info('Retrieved the test set')

        # Retrieve the Neural Network
        nn_model_naive = load_model(self.input()['nn_files']['nn_model_file'].path)

        logger.info('Loaded the Neural Network')

        # Retrieve the SVM
        svm_model_naive = joblib.load(self.input()['svm_model_file'].path)

        logger.info('Loaded the SVM')

        # Retrieve the Decision Tree
        dtc_model_naive = joblib.load(self.input()['dtc_model_file'].path)

        logger.info('Loaded the Decision Tree')

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

        # Create the metrics csv path directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)

        # Append the data to metrics.csv
        with open(self.output().path, 'a') as f:
            # The header gets written only if the csv is empty
            metrics_df.to_csv(f, mode='a', header=f.tell()==0, index=False, lineterminator='\n')

        logger.info('Appended to csv the performance evaluation for each model')
        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(self.metrics_csv)
    


class Completeness(luigi.Task):
    """
    Task to check data completeness by verifying the presence of missing values on the transformed dataset.
    
    Parameters:
    - input-csv: Path to the input csv file containing raw wine data. Default: 'datasets/winetype.csv'
    """

    input_csv = luigi.Parameter(default=default_paths['input_csv'])
    _completion_flag = InMemoryTarget()


    def requires(self):
        # winetype_transformed.csv is needed
        return DataTransformation(input_csv=self.input_csv)


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Read winetype_transformed.csv
        df = pd.read_csv(self.input().path)

        logger.info('Retrieved the transformed dataset')

        # Test the data quality completeness measure on the set
        missing_values, completeness_ratio = dq.completeness_test(df)

        logger.info('============================================================')
        logger.info('COMPLETENESS - MISSING VALUES DISTRIBUTION RESULTS:')
        logger.info('============================================================')
        logger.info(f'\n{missing_values}\n')
        # Info about the ratio of missing values for each feature in the dataset in %
        logger.info('============================================================')
        logger.info('COMPLETENESS - MISSING VALUES RATIO:')
        logger.info('============================================================')
        logger.info(f'\n{completeness_ratio}\n\n')
        
        # Missing values sum
        missing_values_sum = missing_values.sum()

        logger.info(f'Missing values sum: {missing_values_sum}')

        # Completeness check
        if missing_values_sum < dq_count_threshold:
            self._completion_flag.completed = True # The task is now completed
            logger.info('Completeness check passed')
        else:
            logger.error('Completeness check failed')

        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        # Return the custom target instead of a file-based target
        return self._completion_flag



class Consistency(luigi.Task):
    """
    Task to check data consistency by verifying the presence of outliers and inconsistent values on the transformed dataset.
    
    Parameters:
    - input-csv: Path to the input csv file containing raw wine data. Default: 'datasets/winetype.csv'
    """

    input_csv = luigi.Parameter(default=default_paths['input_csv'])
    _completion_flag = InMemoryTarget()


    def requires(self):
        # This task depends on winetype_transformed.csv and the completion of Completeness
        return {'transformed_csv': DataTransformation(input_csv=self.input_csv),
                'completeness_check': Completeness(input_csv=self.input_csv)}


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Read winetype_transformed.csv
        df = pd.read_csv(self.input()['transformed_csv'].path)

        logger.info('Retrieved the transformed dataset')

        # Test the data quality consistency measure on the set
        inconsistent_values_default, inconsistent_values_bounded_std, inconsistent_values_bounded_iqr, outliers_std, outliers_iqr, std_bounds, iqr_bounds = dq.consistency_test(df)

        # Counter for outliers w.r.t the domain (default ranges)
        default_counter = 0
        
        # Counter for outliers w.r.t the mean and std method
        std_counter = 0
        
        # Counter for outliers w.r.t the IQR method
        iqr_counter = 0

        logger.info('============================================================')
        logger.info('CONSISTENCY - INCONSISTENT VALUES DEFAULT RESULTS:')
        logger.info('============================================================')
        # For each feature it has to be 0 (PASSED)
        for feature, count in inconsistent_values_default.items():
            logger.info(f'{feature}: {"PASSED" if count == 0 else count}')
            default_counter += count
        logger.info('')
        logger.info('============================================================')
        logger.info('CONSISTENCY - STD BOUNDS RESULTS:')
        logger.info('============================================================')
        for feature, (lower_bound, upper_bound) in std_bounds.items():
            logger.info(f"{feature}: ({lower_bound}, {upper_bound})")
        logger.info('')
        logger.info('============================================================')
        logger.info('CONSISTENCY - IQR BOUNDS RESULTS:')
        logger.info('============================================================')
        for feature, (lower_bound, upper_bound) in iqr_bounds.items():
            logger.info(f"{feature}: ({lower_bound}, {upper_bound})")
        logger.info('')
        logger.info('============================================================')
        logger.info('CONSISTENCY - INCONSISTENT VALUES STD BOUNDS RESULTS:')
        logger.info('============================================================')
        # For each feature it has to be 0 (PASSED)
        for feature, count in inconsistent_values_bounded_std.items():
            logger.info(f'{feature}: {"PASSED" if count == 0 else count}')
            std_counter += count
        logger.info('')
        logger.info('============================================================')
        logger.info('CONSISTENCY - INCONSISTENT VALUES IQR BOUNDS RESULTS:')
        logger.info('============================================================')
        # For each feature it has to be 0 (PASSED)
        for feature, count in inconsistent_values_bounded_iqr.items():
            logger.info(f'{feature}: {"PASSED" if count == 0 else count}')
            iqr_counter += count
        logger.info('')
        # Info about the outliers found using range calculation with mean and std and bound check
        logger.info('============================================================')
        logger.info('CONSISTENCY - OUTLIERS STD INFO RESULTS:')
        logger.info('============================================================')
        for feature, info in outliers_std.items():
            logger.info(f'Feature: {feature}')
            logger.info(f"Count of outliers: {info['count']}")
            logger.info(f"Rows with outliers: {info['rows']}\n") # Add +2
        logger.info('')
        # Info about the outliers found using interquartile range
        logger.info('============================================================')
        logger.info('CONSISTENCY - OUTLIERS IQR INFO RESULTS:')
        logger.info('============================================================')
        for feature, info in outliers_iqr.items():
            logger.info(f'Feature: {feature}')
            logger.info(f"Count of outliers: {info['count']}")
            logger.info(f"Rows with outliers: {info['rows']}\n") # Add +2
        logger.info('\n')

        logger.info(f'Count of outliers w.r.t the domain: {default_counter}')

        logger.info(f'Count of outliers w.r.t the mean and std method: {std_counter}')

        logger.info(f'Count of outliers w.r.t the IQR method: {iqr_counter}')

        # Domain consistency check
        if default_counter < dq_count_threshold:
            # Mean and std consistency check
            if std_counter < dq_count_threshold:
                # IQR consistency check
                if iqr_counter < dq_count_threshold:
                    logger.info('All consistency checks passed')
                    self._completion_flag.completed = True # The task is now completed
                else:
                    logger.error('Consistency IQR check failed')
            else:
                logger.error('Consistency mean and std check failed')
        else:
            logger.error('Consistency domain check failed')

        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return self._completion_flag
    


class Uniqueness(luigi.Task):
    """
    Task to check data uniqueness by verifying the presence of duplicate values on the transformed dataset.
    
    Parameters:
    - input-csv: Path to the input csv file containing raw wine data. Default: 'datasets/winetype.csv'
    """

    input_csv = luigi.Parameter(default=default_paths['input_csv'])
    _completion_flag = InMemoryTarget()


    def requires(self):
        # This task depends on winetype_transformed.csv and the completion of Consistency
        return {'transformed_csv': DataTransformation(input_csv=self.input_csv),
                'consistency_check': Consistency(input_csv=self.input_csv)}


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Read winetype_transformed.csv
        df = pd.read_csv(self.input()['transformed_csv'].path)

        logger.info('Retrieved the transformed dataset')

        # Test the data quality uniqueness measure on the set
        unique_values, duplicate_values, duplicate_sum, unique_ratio = dq.uniqueness_test(df)

        # Check that every column has at least 2 values
        only_useful_columns = True

        logger.info('============================================================')
        logger.info('UNIQUENESS - UNIQUE VALUES DISTRIBUTION RESULTS:')
        logger.info('============================================================')
        # Gives info about how different the data is in the set, while also
        # verifying if it's meaningful (EX. type has to be 2 since its a boolean)
        logger.info(f'\n{unique_values}\n')
        for _, count in unique_values.items():
            if count == 1:
                # A useless column (no additional information) is in the dataset, since it only has 1 value
                only_useful_columns = False
                break
        # Info about the ratio for each feature of their uniqueness in %
        logger.info('============================================================')
        logger.info('UNIQUENESS - UNIQUE VALUES RATIO RESULTS:')
        logger.info('============================================================')
        logger.info(f'\n{unique_ratio}\n')
        # Info about the distribution of duplicated values
        logger.info('============================================================')
        logger.info('UNIQUENESS - DUPLICATED VALUES DISTRIBUTION RESULTS:')
        logger.info('============================================================')
        logger.info(f'{duplicate_values}\n')
        # Info about the sum (total number) of duplicated values
        logger.info('============================================================')
        logger.info('UNIQUENESS - DUPLICATED VALUES SUM RESULTS:')
        logger.info('============================================================')
        logger.info(f'Duplicated Records: {duplicate_sum}\n\n')

        # Check that every column is useful (at least 2 values)
        if only_useful_columns:
            # Duplicated records check
            if duplicate_sum < dq_count_threshold:
                logger.info('All uniqueness checks passed')
                self._completion_flag.completed = True # The task is now completed
            else:
                logger.error('Uniqueness duplicated records check failed')
        else:
            logger.error('A uniqueness check failed: not every column has at least 2 values')

        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return self._completion_flag



class Accuracy(luigi.Task):
    """
    Task to check data accuracy by verifying the correctness of data types on the transformed dataset.
    
    Parameters:
    - input-csv: Path to the input csv file containing raw wine data. Default: 'datasets/winetype.csv'
    """

    input_csv = luigi.Parameter(default=default_paths['input_csv'])
    _completion_flag = InMemoryTarget()


    def requires(self):
        # This task depends on winetype_transformed.csv and the completion of Uniqueness
        return {'transformed_csv': DataTransformation(input_csv=self.input_csv),
                'uniqueness_check': Uniqueness(input_csv=self.input_csv)}


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # Read winetype_transformed.csv
        df = pd.read_csv(self.input()['transformed_csv'].path)

        logger.info('Retrieved the transformed dataset')

        # Test the data quality accuracy measure on the set
        accuracy_results = dq.accuracy_test(df)

        # Check that every column has the correct type
        correct_types = True

        logger.info('============================================================')
        logger.info('ACCURACY - CORRECT TYPE RESULTS:')
        logger.info('============================================================')
        # For each feature it has to be PASSED
        for feature, is_correct in accuracy_results.items():
            logger.info(f'{feature}: {"PASSED" if is_correct else "FAILED"}')
            if not is_correct:
                correct_types = False
        logger.info('')

        # Accuracy check
        if correct_types:
            logger.info('Accuracy check passed')
            self._completion_flag.completed = True # The task is now completed
        else:
            logger.error('Accuracy check failed')

        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return self._completion_flag



class FullPipeline(luigi.WrapperTask):
    """
    A wrapper task to run the full pipeline with default parameters.

    This is used for executing every single task of the pipeline.
    """
    
    def requires(self):
        return [Accuracy(), PerformanceEval()]