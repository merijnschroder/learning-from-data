from sklearn.base import BaseEstimator
from transformers import TFAutoModelForSequenceClassification
import tensorflow as tf

from lfd.models.classifier_base import BaseLMClassifier
from lfd.models.data import Data


class LanguageModelClassifier(BaseLMClassifier):
    '''A Language Model classifier taken from HuggingFace'''

    _classifier_name: str = 'PLM'
    _classifier: BaseEstimator

    def __init__(self,
                 lm: str,
                 model_verbosity: int = 0,
                 training_verbosity: int = 1,
                 learning_rate=0.00005) -> None:
        self.model_verbosity = model_verbosity
        self.training_verbosity = training_verbosity

        self.lm = lm
        self._classifier_name += f'_{lm}'

        self._classifier = self.lm_create(
            num_labels=1,
            learning_rate=learning_rate
        )
        super().__init__()

    def lm_create(self, num_labels: int,
                     learning_rate: float) -> None:
        '''Create the PLM model and set learning rate'''
        model = TFAutoModelForSequenceClassification.from_pretrained(
            self.lm, num_labels=num_labels
        )
        # loss_function = tf.keras.losses.CategoricalCrossentropy(
        loss_function = tf.keras.losses.BinaryCrossentropy(
            from_logits=True
        )
        optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(
            loss=loss_function, optimizer=optim, metrics=['accuracy'])

        return model

    def grid_search(self, data: Data):
        param_grid = {
            'learning_rate': [0.00005, 0.00000005, 0.0000000001],
            'epochs': [1, 3, 5],
            'batch_size': [0.005, 0.05, 0.5],
        }
        self._grid_search(data, param_grid)
