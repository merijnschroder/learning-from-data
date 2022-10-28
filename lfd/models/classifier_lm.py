import logging
import tensorflow as tf
from lfd.models.classifier_base import BaseClassifier
from lfd.models.data import Data
from transformers import PreTrainedModel, TFAutoModelForSequenceClassification
from typing_extensions import override


class LanguageModelClassifier(BaseClassifier):
    '''A Language Model classifier taken from HuggingFace'''

    classifier_id: str
    _model_name: str
    _classifier_name: str = 'PLM'
    _classifier: PreTrainedModel
    _model_verbosity: int
    _training_verbosity: int
    _is_trained: bool = False

    def __init__(
        self,
        model_name: str,
        model_verbosity: int = 0,
        training_verbosity: int = 1,
        learning_rate=0.00005
    ) -> None:
        self._model_name = model_name
        self._classifier_name += f'_{model_name}'
        self._model_verbosity = model_verbosity
        self._training_verbosity = training_verbosity
        self._classifier = \
            self.create_classifier(num_labels=1, learning_rate=learning_rate)
        super().__init__()

    def create_classifier(self, num_labels: int,
                          learning_rate: float) -> PreTrainedModel:
        '''Create the PLM model and set learning rate'''
        model: PreTrainedModel = \
            TFAutoModelForSequenceClassification.from_pretrained(
                self._model_name, num_labels=num_labels)
        loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            loss=loss_function, optimizer=optim, metrics=['accuracy'])
        return model

    def evaluate_dev(self, data: Data):
        self._evaluate(data.tokens_dev, data.y_dev_bin)

    def evaluate_test(self, data: Data):
        self._evaluate(data.tokens_test, data.y_test_bin)

    def grid_search(self, data: Data):
        param_grid = {
            'learning_rate': [0.00005, 0.00000005, 0.0000000001],
            'epochs': [1, 3, 5],
            'batch_size': [0.005, 0.05, 0.5],
        }
        self._grid_search(data, param_grid)

    @override
    def _train(self, data: Data):
        validation_data = (data.tokens_dev, data.y_dev_bin)
        self._classifier.fit(
            data.tokens_train,
            data.y_train_bin,
            verbose=self._training_verbosity,
            epochs=1,
            batch_size=16,
            validation_data=validation_data
        )

    @override
    def _evaluation_prediction(self, x_test):
        logging.info('Start evaluating %s on the data points',
                     self.classifier_name)
        output = self._classifier.predict(x_test)["logits"]
        return tf.round(tf.nn.sigmoid(output))

    @override
    def _grid_search_fitting(self, grid_search, data):
        grid_search.fit(data.tokens_train, data.y_train_bin)
