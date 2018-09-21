from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
import numpy as np
import argparse


def individual_model(keys, labels, config):
    model = Sequential()
    model.add(Dense(config['0'], input_dim=1))
    model.add(LeakyReLU())
    for i in range(1000):
        if str(i) not in config:
            break
        model.add(Dense(int(config[str(i)])))
        model.add(LeakyReLU())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=[
                  max_absolute_error, 'mse', 'mae'])
    model.fit(keys, labels, epochs=64, batch_size=32, verbose=1)
    # model.compile(optimizer='adam', loss=mean_fourth_error,
    #               metrics=[max_absolute_error, 'mse', 'mae'])
    # model.fit(keys, labels, epochs=1, batch_size=32, verbose=1)
    # model.compile(optimizer='adam', loss=max_absolute_error,
    #               metrics=[max_absolute_error, 'mse', 'mae'])
    # model.fit(keys, labels, epochs=1, batch_size=32, verbose=1)
    # model.compile(optimizer='adam', loss=max_absolute_error,
    #               metrics=[max_absolute_error, 'mse', 'mae'])
    # model.fit(keys, labels, epochs=1, batch_size=128, verbose=1)
    return model


def max_absolute_error_model(keys, labels, model):
    pred_labels = model.predict(keys)
    # for some inane reason, Python hangs if I use Keras functions here???
    return np.max(np.abs(labels - pred_labels[:, 0]))


def max_absolute_error(y_true, y_pred):
    return K.max(K.abs(y_true - y_pred))


def mean_fourth_error(y_true, y_pred):
    return K.mean(K.square(K.square(y_pred - y_true)), axis=-1)


def mean_sixth_error(y_true, y_pred):
    return K.mean(K.square(K.square(K.square(y_pred - y_true))), axis=-1)


def plot(keys, labels, model):
    pred_labels = model.predict(keys)
    import matplotlib.pyplot as plt
    plt.plot(keys, labels)
    plt.plot(keys, pred_labels)
    plt.show()


def select_next_model(pred_label, max_pred, model_count):
    index = (K.flatten(pred_label) / max_pred) * model_count
    x = K.clip(index, 0, model_count - 1)
    return K.clip(index, 0, model_count - 1)


def max_overestimate(pred_label, label):
    return max(K.get_value(K.max(K.flatten(pred_label) - K.flatten(label))), 0)


def max_underestimate(pred_label, label):
    return max(K.get_value(K.max(K.flatten(label) - K.flatten(pred_label))), 0)


def train(toml_file, data_file):
    import toml
    with open(toml_file) as f:
        text = f.read()
    t = toml.loads(text)

    keys = np.loadtxt(data_file, dtype=np.float32)
    keys = keys[:, np.newaxis]
    labels = np.arange(len(keys), dtype=np.float32)

    model = individual_model(keys, labels, t['model'])

    first_layer = model.layers[0]

    btree_indices = [[] for _ in range(t['model']['btree_count'])]

    pred_labels = model.predict(keys)

    print('select models')
    selected_models = select_next_model(
        pred_labels, labels[-1], t['model']['btree_count'])

    selected_models = K.eval(selected_models)

    for i in range(len(pred_labels)):
        selected_model = int(selected_models[i])
        btree_indices[selected_model].append(i)

    print("labels", pred_labels[:5])

    return model, btree_indices


def save(filename, model, btree_indices):
    with open(filename, 'w') as f:
        j = 0
        for layer in model.layers:
            weight_list = layer.get_weights()
            if not weight_list:
                continue
            f.write("layer{} = [".format(j))
            for array in weight_list:
                f.write('[')
                flattened = K.eval(K.flatten(array))
                for item in flattened:
                    f.write("{0:f}, ".format(item))
                f.write('], ')
            f.write("]\n")
            j += 1
        f.write("btree_indices = {}\n".format(btree_indices))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--index', required=True,
        help="File containing newline separated numbers representing keys")

    parser.add_argument(
        '--config', required=True,
        help="TOML file describing model architecture",
    )

    parser.add_argument(
        '--save', required=True,
        help="File in which to save trained model",
    )

    args = parser.parse_args()

    model, btree_indices = train(args.config, args.index)

    save("/Users/mike/trashout.toml", model, btree_indices)


if __name__ == '__main__':
    main()
