"""
TODO docstring

Cavaleri Matteo - 875050
Gargiulo Elio - 869184
Piacente Cristian - 866020
"""

import luigi
import logging

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

        # TODO add logic

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

        # TODO add logic

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

        # TODO add logic

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

        # TODO add logic

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


    def requires(self):
        # winetype_pca_train.csv and winetype_pca_test.csv are needed
        return SplitDataset(input_csv=self.input_csv)
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # TODO add logic

        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(self.nn_model_file)
    


class SVMModel(luigi.Task):
    """
    TODO docstring
    """

    input_csv = luigi.Parameter(default=default_paths['input_csv'])
    svm_model_file = luigi.Parameter(default=default_paths['svm_model_file'])


    def requires(self):
        # winetype_pca_train.csv and winetype_pca_test.csv are needed
        return SplitDataset(input_csv=self.input_csv)
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # TODO add logic

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
        # winetype_pca_train.csv and winetype_pca_test.csv are needed
        return SplitDataset(input_csv=self.input_csv)
    

    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # TODO add logic

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
        # nn_model, svm_model, dtc_model, winetype_pca_train.csv and winetype_pca_test.csv are needed
        return {'nn_model_file': NNModel(input_csv=self.input_csv),
                'svm_model_file': SVMModel(input_csv=self.input_csv),
                'dtc_model_file': DTCModel(input_csv=self.input_csv),
                'splitted_dataset_csv': SplitDataset(input_csv=self.input_csv)}


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # TODO add logic

        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return luigi.LocalTarget(self.metrics_csv)
    


class Consistency(luigi.Task):
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

        # Consistency check passed
        self._completion_flag.completed = True

        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        # Return the custom target instead of a file-based target
        return self._completion_flag



class Coherence(luigi.Task):
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

        # Coherence check passed
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
        # This task depends on the completion of Coherence
        return Coherence(input_csv=self.input_csv)


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # TODO add logic

        # Accuracy check passed
        self._completion_flag.completed = True

        logger.info(f'Finished task {self.__class__.__name__}')


    def output(self):
        return self._completion_flag



class Integrity(luigi.Task):
    """
    TODO docstring
    """

    input_csv = luigi.Parameter(default=default_paths['input_csv'])
    _completion_flag = InMemoryTarget()


    def requires(self):
        # This task depends on the completion of Accuracy
        return Accuracy(input_csv=self.input_csv)


    def run(self):
        logger.info(f'Started task {self.__class__.__name__}')

        # TODO add logic

        # Integrity check passed
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
        return [Integrity(), PerformanceEval()]