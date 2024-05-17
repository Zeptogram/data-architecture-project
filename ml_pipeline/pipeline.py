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

from utils.evaluation import get_global_metrics, get_confidence_intervals

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
    TODO docstring
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
    TODO docstring
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
    TODO docstring
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
    TODO docstring
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
    TODO docstring
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
        history_naive = nn_model_naive.fit(X_train, y_train, epochs=10, batch_size=32)

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
    TODO docstring
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
    TODO docstring
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
    TODO docstring
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
    TODO docstring
    """

    input_csv = luigi.Parameter(default=default_paths['input_csv'])
    _completion_flag = InMemoryTarget()


    def requires(self):
        # winetype_transformed.csv is needed
        return DataTransformation(input_csv=self.input_csv)


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # TODO add logic

        # Completeness check passed
        self._completion_flag.completed = True

        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        # Return the custom target instead of a file-based target
        return self._completion_flag



class Consistency(luigi.Task):
    """
    TODO docstring
    """

    input_csv = luigi.Parameter(default=default_paths['input_csv'])
    _completion_flag = InMemoryTarget()


    def requires(self):
        # This task depends on the completion of Completeness
        return Completeness(input_csv=self.input_csv)


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # TODO add logic

        # Consistency check passed
        self._completion_flag.completed = True

        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return self._completion_flag
    


class Uniqueness(luigi.Task):
    """
    TODO docstring
    """

    input_csv = luigi.Parameter(default=default_paths['input_csv'])
    _completion_flag = InMemoryTarget()


    def requires(self):
        # This task depends on the completion of Consistency
        return Consistency(input_csv=self.input_csv)


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # TODO add logic

        # Uniqueness check passed
        self._completion_flag.completed = True

        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return self._completion_flag



class Accuracy(luigi.Task):
    """
    TODO docstring
    """

    input_csv = luigi.Parameter(default=default_paths['input_csv'])
    _completion_flag = InMemoryTarget()


    def requires(self):
        # This task depends on the completion of Uniqueness
        return Uniqueness(input_csv=self.input_csv)


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # TODO add logic

        # Accuracy check passed
        self._completion_flag.completed = True

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