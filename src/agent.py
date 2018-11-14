import logging
import keras
import numpy as np
import tensorflow as tf

_log = logging.getLogger('MLProject')


class NNAgent:
    def __init__(self, state_space_shape, action_space_size):
        """
        Creates a NNAgent and initializes its neural network with random weights.
        :param state_space_shape: the shape of the state space
        :param action_space_size: the number of actions in the action space
        """
        hidden_layer_size = np.prod(state_space_shape)
        self.state_space_shape = state_space_shape
        self.action_space_size = action_space_size

        # Create a unique tensorflow graph and session for this agent only
        self._tf_graph = tf.Graph()
        self._tf_sess = tf.Session(graph=self._tf_graph)

        # Set this model's graph and session as the default one, so the model is assigned to them
        with self._tf_graph.as_default(), self._tf_sess.as_default():
            self._model = tf.keras.Sequential([
                tf.keras.layers.Reshape((hidden_layer_size,), input_shape=state_space_shape),  # Flatten the state
                # tf.keras.layers.Flatten(),                       # Note: Reshape and Flatten yield different results
                tf.keras.layers.Dense(hidden_layer_size),         # First hidden layer with as many neurons as state pixels
                # tf.keras.layers.Dense(hidden_layer_size),         # Second hidden layer
                # tf.keras.layers.Dense(hidden_layer_size // 2),    # Third hidden layer
                tf.keras.layers.Dense(action_space_size),         # Output layer
                tf.keras.layers.Activation('softmax')             # Softmax activation
            ])

    def act(self, state: np.ndarray):
        """
        Acts on a state. Returns the best action to take based on the given state.
        :param state: the state, as an ndarray
        :return: an ndarray with the same number of elements as the action space size
        """
        with self._tf_graph.as_default(), self._tf_sess.as_default():
            # We don't use batching, so we wrap the state into a batch array which we
            # feed into model.predict(). Then we un-wrap the prediction from its batch array
            state_batch = np.reshape(state, (1, *state.shape))
            action = self._model.predict(state_batch, batch_size=1)[0]
        return action

    def save_model(self, filename):
        with self._tf_graph.as_default(), self._tf_sess.as_default():
            self._model.save(filename)

    def load_model(self, filename):
        with self._tf_graph.as_default(), self._tf_sess.as_default():
            self._model = keras.models.load_model(filename)

    def get_weights(self):
        """
        Returns the weights of the NN for this agent.
        :return:
        """
        with self._tf_graph.as_default(), self._tf_sess.as_default():
            weights = self._model.get_weights()
        return weights

    def set_weights(self, new_weights):
        """
        Sets new weights for the NN of this agent.
        :param new_weights:
        :return:
        """
        with self._tf_graph.as_default(), self._tf_sess.as_default():
            self._model.set_weights(new_weights)


if __name__ == '__main__':
    def vectofixedstr(vec, presicion=8):
        ret = []
        for el in vec:
            ret.append('{:.{}f}'.format(el, presicion))
        return '[' + ' '.join(ret) + ']'

    agent = NNAgent(state_space_shape=(10,), action_space_size=5)
    action = agent.act(np.array([1, 3, 5, 7, 2, 4, 6, 8, 9, 9]))
    print(vectofixedstr(action))
    print(np.argmax(action))
