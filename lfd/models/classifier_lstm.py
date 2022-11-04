import logging

from lfd.models.classifier_base import BaseClassifier
from lfd.models.data import Data

from keras.initializers.initializers_v2 import Constant
from keras.layers import LSTM, Embedding, Bidirectional
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD, Adagrad, Adam, Adamax
import tensorflow as tf
import numpy as np
import json


class LSTMClassifier(BaseClassifier):
    '''An LSTM classifier'''

    classifier_id: str
    _classifier_name: str = 'LSTM'
    _classifier: Sequential
    _is_trained: bool = False

    def __init__(
        self,
        data: Data,
        optimizer='Adam',
        learning_rate=1e-05,
        non_pretrained_embeddings=False,
        trainable_embeddings=True,
        architecture='300.3r',
        bidirectional=True,
        epochs=50,
        batch_size=16
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size

        logging.info(
            'The settings that are used '
            'for the LSTM model are:\n'
            f'{"optimizer:":<29}{optimizer}\n'
            f'{"learning rate:":<29}{learning_rate}\n'
            f'{"non pretrained embeddings:":<29}'
            f'{non_pretrained_embeddings}\n'
            f'{"trainable embeddings:":<29}{trainable_embeddings}\n'
            f'{"architecture:":<29}{architecture}\n'
            f'{"bidirectional:":<29}{bidirectional}\n'
            f'{"epochs:":<29}{epochs}\n'
            f'{"batch size:":<29}{batch_size}'
        )
        self._classifier = self.create_model(
                data=data,
                optimizer=optimizer,
                learning_rate=learning_rate,
                non_pretrained_embeddings=non_pretrained_embeddings,
                trainable_embeddings=trainable_embeddings,
                architecture=architecture,
                bidirectional=bidirectional
        )
        super().__init__()

    def grid_search(self, data: Data):
        param_grid = {
            'optimizer': ['SGD', 'Adam', 'Adagrad', 'Adamax'],
            'learning_rate': [0.5, 0.05, 0.00005],
            'non_pretrained_embeddings': [False, True],
            'trainable_embeddings': [True, False],
            'architecture': '3',
            'bidirectional': [False, True]
        }
        self._grid_search(data, param_grid)

    def create_model(self,
        data,
        optimizer,
        learning_rate,
        non_pretrained_embeddings,
        trainable_embeddings,
        architecture,
        bidirectional
    ) -> Sequential:
        '''Create the Keras model to use'''
        output_layer_size = 1

        # Define settings
        loss_function = 'binary_crossentropy'
        optim = self._get_optimizer(optimizer, learning_rate)

        # Create an embedding matrix from a given file
        embedding_file = 'lfd/embeddings/' + \
                         'glove.twitter.filtered.100d' + \
                         '.json'
        self.load_embeddings(embedding_file, data)

        # Take embedding dim and size from emb_matrix
        embedding_dim = len(self.emb_matrix[0])
        num_tokens = len(self.emb_matrix)

        # Build the model
        model = Sequential()

        if non_pretrained_embeddings:
            embeddings_initializer = "uniform"
        else:
            embeddings_initializer = Constant(self.emb_matrix)

        model.add(Embedding(
            num_tokens, embedding_dim,
            embeddings_initializer=embeddings_initializer,
            trainable=trainable_embeddings
        ))

        # Add the LSTM layers.
        parsed_architecture = self._parse_architecture(architecture)
        for i, layer in enumerate(parsed_architecture):
            if not layer[2]:
                LSTM_layer = LSTM(units=layer[0], dropout=layer[1])
            else:
                LSTM_layer = LSTM(units=layer[0], recurrent_dropout=layer[1])

            if i < len(parsed_architecture) - 1:
                LSTM_layer.return_sequences = True

            # Make the layer bidirectional.
            if bidirectional:
                LSTM_layer = Bidirectional(LSTM_layer)

            model.add(LSTM_layer)

        # Ultimately, end with dense layer with softmax
        model.add(Dense(
            input_dim=embedding_dim, units=output_layer_size,
            activation="sigmoid"))

        # Compile model using our settings, check for accuracy
        model.compile(loss=loss_function, optimizer=optim,
                      metrics=['accuracy'])
        return model

    @staticmethod
    def _parse_architecture(
            architecture_arg: str) -> 'list[tuple[int, float, bool]]':
        result: list[tuple[int, float, bool]] = []
        for arg in architecture_arg.split('-'):
            arg_split = arg.split('.')
            units = int(arg_split[0])
            dropout = 0

            # Determine whether the dropout is recurrent.
            recurrent = False
            if len(arg_split) > 1:
                if str(arg_split[1]).endswith('r'):
                    recurrent = True
                    arg_split[1] = str(arg_split[1]).replace('r', '')
                dropout = float('0.' + arg_split[1])

            result.append((units, dropout, recurrent))
        return result

    @staticmethod
    def _get_optimizer(optimizer_arg: str, learning_rate: float):
        if optimizer_arg == 'SGD':
            return SGD(learning_rate)
        if optimizer_arg == 'Adam':
            return Adam(learning_rate)
        if optimizer_arg == 'Adagrad':
            return Adagrad(learning_rate)
        if optimizer_arg == 'Adamax':
            return Adamax(learning_rate)
        raise Exception(f'Invalid optimizer {optimizer_arg}')

    def load_embeddings(self, embeddings, data):
        '''Load and fit the embeddings given a file.'''
        self.emb_matrix = self._get_emb_matrix(
            data.voc,
            self._read_embeddings(embeddings)
        )

    def _read_embeddings(self, embeddings_file):
        '''Read in word embeddings from file and save as numpy array'''
        embeddings = json.load(open(embeddings_file, 'r', encoding='utf-8'))
        return {word: np.array(embeddings[word]) for word in embeddings}

    def _get_emb_matrix(self, voc, emb):
        '''Get embedding matrix given vocab and the embeddings'''
        num_tokens = len(voc) + 2
        word_index = dict(zip(voc, range(len(voc))))
        # Bit hacky, get embedding dimension from the word "the"
        embedding_dim = len(emb["the"])
        # Prepare embedding matrix to the correct size
        embedding_matrix = np.zeros((num_tokens, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = emb.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        # Final matrix with pretrained embeddings that we can feed to embedding
        # layer
        return embedding_matrix

    def evaluate_dev(self, data: Data):
        self._evaluate(data.get_x_dev(), data.get_y_dev(encoded=True))

    def evaluate_test(self, data: Data):
        self._evaluate(data.get_x_test(), data.get_y_test(encoded=True))

    def _train(self, data):
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=3)
        self._classifier.fit(  # type: ignore
            data.get_x_train(),
            data.get_y_train(encoded=True),
            verbose=1,
            epochs=self.epochs,
            callbacks=[callback],
            batch_size=self.batch_size,
            validation_data=(data.get_x_dev(), data.get_y_dev(encoded=True)),
        )
        self.encoder = data.encoder

    def _evaluation_prediction(self, x_test):
        logging.info('Start evaluating %s on %d data points',
                     self.classifier_name, x_test.shape[0])
        # return self._classifier.predict(x_test)  # type: ignore
        return self.encoder.inverse_transform(self._classifier.predict(x_test))  # type: ignore

    def _grid_search_fitting(self, grid_search, data):
        # grid_search.fit(data.get_x_train().toarray(), data.get_y_train(encoded=True))
        grid_search.fit(data.get_x_train(), data.get_y_train(encoded=True))
