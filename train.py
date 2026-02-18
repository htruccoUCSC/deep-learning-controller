import time

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import image_size
from models.basic_model import BasicModel
from models.model import Model
from preprocess import get_datasets

input_shape = (image_size[0], image_size[1], 3)
categories_count = 3

models = {
    'basic_model': BasicModel,
}

RUN_HYPERPARAM_SEARCH = True

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize = (24, 6))
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, 'b', label = 'Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, 'b', label = 'Training Loss')
    plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.show()

def get_best_val_accuracy(history):
    val_acc = history.history.get('val_accuracy', [])
    if not val_acc:
        return 0.0, 0
    best_epoch = int(np.argmax(val_acc)) + 1
    return float(np.max(val_acc)), best_epoch

def run_hyperparam_search(
    train_dataset,
    validation_dataset,
    input_shape,
    categories_count,
    epochs,
    callbacks,
):
    dropout_options = [
        (None, None),
        (0.25, None),
        (None, 0.5),
        (0.25, 0.5),
    ]
    learning_rates = [1e-3, 5e-4, 1e-4]

    all_configs = [(dropout_pair, lr) for dropout_pair in dropout_options for lr in learning_rates]

    results = []
    best_model = None
    best_history = None
    best_score = -1.0
    best_config = None

    for i, (dropout_pair, learning_rate) in enumerate(all_configs, start=1):
        dropout_after_conv_rate, dropout_after_dense_rate = dropout_pair
        print(f"* Trial {i}/{len(all_configs)}: dropout=({dropout_after_conv_rate},{dropout_after_dense_rate}), "
              f"lr={learning_rate}")

        model = BasicModel(
            input_shape,
            categories_count,
            dropout_after_conv_rate=dropout_after_conv_rate,
            dropout_after_dense_rate=dropout_after_dense_rate,
            learning_rate=learning_rate,
        )

        history = model.train_model(
            train_dataset,
            validation_dataset,
            epochs,
            callbacks=callbacks,
        )

        best_val_acc, best_epoch = get_best_val_accuracy(history)
        results.append({
            "dropout_after_conv": dropout_after_conv_rate,
            "dropout_after_dense": dropout_after_dense_rate,
            "learning_rate": learning_rate,
            "best_val_accuracy": best_val_acc,
            "best_epoch": best_epoch,
        })

        if best_val_acc > best_score:
            best_score = best_val_acc
            best_model = model
            best_history = history
            best_config = results[-1]

    return best_model, best_history, best_config, results

if __name__ == "__main__":
    # if you want to load your model later, you can use:
    # model = Model.load_model("name_of_your_model.keras")
    # to load your history and plot it again, you can use:
    # history = np.load('results/name_of_your_model.npy',allow_pickle='TRUE').item()
    # plot_history(history)
    # 
    # Your code should change the number of epochs
    epochs = 40
    print('* Data preprocessing')
    train_dataset, validation_dataset, test_dataset = get_datasets()
    name = 'basic_model'
    model_class = models[name]
    print('* Training {} for {} epochs'.format(name, epochs))
    model = model_class(input_shape, categories_count)
    model.print_summary()
    early_stop = EarlyStopping(monitor='val_accuracy', patience=6, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=0.00001)
    callbacks = [early_stop, reduce_lr]

    if RUN_HYPERPARAM_SEARCH:
        print('* Hyper-parameter search')
        model, history, best_config, results = run_hyperparam_search(
            train_dataset,
            validation_dataset,
            input_shape,
            categories_count,
            epochs,
            callbacks,
        )

        print('* Best config: {}'.format(best_config))
    else:
        print('* Training {} for {} epochs'.format(name, epochs))
        model = model_class(input_shape, categories_count)
        model.print_summary()
        history = model.train_model(train_dataset, validation_dataset, epochs, callbacks=callbacks)

    print('* Evaluating {}'.format(name))
    model.evaluate(test_dataset)
    print('* Confusion Matrix for {}'.format(name))
    print(model.get_confusion_matrix(test_dataset))
    model_name = '{}_{}_epochs_timestamp_{}'.format(name, epochs, int(time.time()))
    filename = 'results/{}.keras'.format(model_name)
    model.save_model(filename)
    np.save('results/{}.npy'.format(model_name), history)
    print('* Model saved as {}'.format(filename))
    plot_history(history)
