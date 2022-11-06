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
    _classifier_name: str
    _classifier: PreTrainedModel
    _model_verbosity: int
    _training_verbosity: int
    _is_trained: bool = False

    def __init__(
        self,
        model_name: str,
        model_verbosity: int = 0,
        training_verbosity: int = 1,
        learning_rate=1e-5,
        batch_size=8,
        epochs=1
    ) -> None:
        self._model_name = model_name
        self._classifier_name = f'PLM_{model_name}'
        self._model_verbosity = model_verbosity
        self._training_verbosity = training_verbosity

        if model_name == 'GroNLP/hateBERT':
            self.gro_pt = True
        else:
            self.gro_pt = False

        self._classifier = \
            self.create_classifier(num_labels=1, learning_rate=learning_rate)
        self._batch_size = batch_size
        self._epochs = epochs
        bert_tuple = ('bert-base-uncased', 'bert-base-cased')
        if model_name in bert_tuple:
            self._epochs = 3
        super().__init__()

    def create_classifier(self, num_labels: int,
                          learning_rate: float) -> PreTrainedModel:
        '''Create the PLM model and set learning rate'''
        model: PreTrainedModel = \
            TFAutoModelForSequenceClassification.from_pretrained(
                self._model_name, num_labels=num_labels, from_pt=self.gro_pt)
        loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            loss=loss_function, optimizer=optim, metrics=['accuracy'])
        return model

    def evaluate_dev(self, data: Data):
        self._evaluate(
            data.get_x_dev(self._model_name), data.get_y_dev(True))

    def evaluate_test(self, data: Data):
        self._evaluate(
            data.get_x_test(self._model_name), data.get_y_test(True))

    @override
    def _train(self, data: Data):
        validation_data = (
            data.get_x_dev(self._model_name), data.get_y_dev(True))
        self._classifier.fit(
            data.get_x_train(self._model_name),
            data.get_y_train(True),
            verbose=self._training_verbosity,
            epochs=self._epochs,
            batch_size=self._batch_size,
            validation_data=validation_data
        )

    @override
    def _evaluation_prediction(self, x_test):
        logging.info('Start evaluating %s on the data points',
                     self.classifier_name)
        output = self._classifier.predict(x_test)["logits"]
        return tf.round(tf.nn.sigmoid(output))

    @override
    def grid_search(self, data: Data):
        pass

    @override
    def _grid_search_fitting(self, grid_search, data):
        pass