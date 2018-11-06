import keras.models as m


def save_to_file(model, index):
    """
    Saves Keras model to file
    :param model:
    """
    model.save("models/model{}.h5".format(index))


def load_from_file(path):
    """
    Load keras model from file
    :param path: path to file
    :return: the model from file
    """
    return m.load_model(path)
