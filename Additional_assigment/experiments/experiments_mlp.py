from collections import defaultdict
from time import time

from net.callbacks import LoggerUpdater
from net.initializers import *
from net.layers.activations import *
from net.losses import categorical_cross_entropy, mean_squared_error
from net.metrics import LabelAccuracy
from net.model import MlpNet
from net.optimizers import *
from settings import DATA_PATH
from training import Trainer
from utils import plot_val_loss_per_batch, load_data, plot_val_loss, plot_val_loss_repeated, plot_val_accuracy, \
    plot_val_accuracy_repeated, plot_time_bar, plot_val_vs_train_acc, plot_cross_mse_val_loss


def learning_rate_experiment(model_dict, train_dict):
    def exp_generator():
        for optimizer, params in [
            (SGD(learning_rate=0.01), ': lr=0.01'),
            (SGD(learning_rate=0.05), ': lr=0.05'),
            (SGD(learning_rate=0.1), ': lr=0.1'),
            (SGD(learning_rate=0.15), ': lr=0.15'),
            (SGD(learning_rate=0.2), ': lr=0.2')
        ]:
            model_dict['optimizer'] = optimizer
            yield model_dict, train_dict, 'optimizer', optimizer.name + params

    return exp_generator


def batch_size_experiment(model_dict, train_dict):
    def exp_generator():
        for batch_size in [1000, 500, 200, 100, 50, 20, 10]:
            train_dict['batch_size'] = batch_size
            yield model_dict, train_dict, 'batch_size', batch_size

    return exp_generator


def init_range_experiment(model_dict, train_dict):
    def exp_generator():
        for initializer, range in [
            (RangeInit(range=(-0.01, 0.01)), '[-0.01, 0.01]'),
            (RangeInit(range=(-0.1, 0.1)), '[-0.1, 0.1]'),
            (RangeInit(range=(-0.2, 0.2)), '[-0.2, 0.2]'),
            (RangeInit(range=(-0.5, 0.5)), '[-0.5, 0.5]'),
            (RangeInit(range=(-1.0, 1.0)), '[-1.0, 1.0]')
        ]:
            model_dict['initializer'] = initializer
            train_dict['batch_size'] = 30
            yield model_dict, train_dict, 'init_range', range

    return exp_generator


def initializer_experiment(model_dict, train_dict):
    def exp_generator():
        for initializer in [
            Xavier(),
            KaimingHe()
        ]:
            model_dict['initializer'] = initializer
            yield model_dict, train_dict, 'initializer', initializer.name

    return exp_generator


def hidden_layer_experimnt(model_dict, train_dict):
    def exp_generator():
        for hidden_units in [10, 50, 100, 200, 500]:
            model_dict['hidden_units'] = (hidden_units,)
            train_dict['epochs'] = 30
            yield model_dict, train_dict, 'hidden_layer', hidden_units

    return exp_generator


def activation_fun_experiment(model_dict, train_dict):
    def exp_generator():
        for act_fun in [Sigmoid, ReLU, Tanh]:
            model_dict['activation'] = act_fun
            yield model_dict, train_dict, 'activation', act_fun.name

    return exp_generator


def loss_fun_experiment(model_dict, train_dict):
    def exp_generator():
        for loss_fun, loss_name in [
            (categorical_cross_entropy, 'cross_entropy'),
            (mean_squared_error, 'MSE')
        ]:
            model_dict['loss_fun'] = loss_fun
            train_dict['epochs'] = 50
            yield model_dict, train_dict, 'loss_function', loss_name

    return exp_generator


def optimizer_experiment(model_dict, train_dict):
    def exp_generator():
        for optimizer, params in [
            (SGDMomentum(learning_rate=0.1), ': lr=0.1'),
            (NAG(learning_rate=0.01), ': lr=0.01'),
            (Adagrad(), ': rho=0.01'),
            (Adadelta(), ': rho=0.95'),
            (Adam(), ': lr=0.001')
        ]:
            model_dict['optimizer'] = optimizer
            yield model_dict, train_dict, 'optimizer', optimizer.name + params

    return exp_generator


def run_experiment(experiment_generator, out_dir, test_data, plot_loss_batch=False):
    np.random.seed(12345)
    results = defaultdict(list)

    for i, (model_dict, train_dict, exp_name, value) in enumerate(experiment_generator()):
        model = MlpNet(**model_dict)
        trainer = Trainer(model, **train_dict)

        label = f'{exp_name}={value}'
        print(f'{i}. {label}')

        start_time = time()
        trainer.train_loop()
        time_period = time() - start_time

        log_data = trainer.logger.logging_data

        if plot_loss_batch:
            # plot train loss per batch in first epoch
            filename = exp_name + str(value) + '_loss_one_batch'
            plot_val_loss_per_batch(log_data['loss_batch']['train'], filename, out_dir)

        results['model_dict'].append(model_dict)
        results['train_dict'].append(train_dict)
        results['time'].append(time_period)
        results['label'].append(label)
        results['log_data'].append(log_data)

        # calculate accuracy on test data
        acc_metric = LabelAccuracy()
        x_test, y_test = test_data
        accuracy = acc_metric(model.predict_classes(x_test), y_test)
        print('Accuracy on test data: {}'.format(accuracy))

    return results


if __name__ == "__main__":
    train_data, val_data, test_data = load_data(DATA_PATH)

    model_dict = {
        'optimizer': SGD(),
        'initializer': Xavier(),
        'metrics': [LabelAccuracy()],
        'loss_fun': categorical_cross_entropy,
        'activation': Sigmoid,
        'hidden_units': (500,)
    }

    train_dict = {
        'train_data': train_data,
        'val_data': val_data,
        'epochs': 30,
        'batch_size': 30,
        'callbacks': [
            # ModelDump(output_dir=out_dir),
            # SaveBestModel(output_dir=out_dir),
            LoggerUpdater()
        ]
    }
    '''
    # Experiment - LEARNING RATE
    results = run_experiment(learning_rate_experiment(model_dict, train_dict), out_dir='learning_rate',
                             test_data=test_data, plot_loss_batch=True)
    plot_val_loss(results, dirname='learning_rate')
    plot_val_accuracy(results, dirname='learning_rate')
    plot_time_bar(results, dirname='learning_rate')


    # Experiment - BATCH SIZE
    results = run_experiment(batch_size_experiment(model_dict, train_dict), out_dir='batch_size',
                             test_data=test_data, plot_loss_batch=True)
    plot_val_loss(results, dirname='batch_size')
    plot_val_accuracy(results, dirname='batch_size')
    plot_time_bar(results, dirname='batch_size')

    # Experiment - INIT RANGE
    results = run_experiment(init_range_experiment(model_dict, train_dict), out_dir='init_range',
                             test_data=test_data)
    plot_val_loss(results, dirname='init_range')
    plot_val_accuracy(results, dirname='init_range')

    # Experiment - HIDDEN LAYER SIZE
    results = run_experiment(hidden_layer_experimnt(model_dict, train_dict), out_dir='hidden_layer',
                             test_data=test_data)
    plot_val_loss(results, dirname='hidden_layer')
    plot_val_accuracy(results, dirname='hidden_layer')
    plot_val_vs_train_acc(results, dirname='hidden_layer')
    plot_time_bar(results, dirname='hidden_layer')
    
    # Experiment - LOSS FUNCTION
    results = run_experiment(loss_fun_experiment(model_dict, train_dict), out_dir='loss_fun',
                             test_data=test_data)

    plot_cross_mse_val_loss(results, dirname='loss_fun')
    plot_val_accuracy(results, dirname='loss_fun')
    plot_time_bar(results, dirname='loss_fun')
'''
    # Experiment - initializers
    repeat = 10
    results = [run_experiment(initializer_experiment(model_dict, train_dict), out_dir='initializer',
                              test_data=test_data) for i in range(repeat)]

    plot_val_loss_repeated(results, dirname='initializer')
    plot_val_accuracy_repeated(results, dirname='initializer')
'''
    # Experiment - ACTIVATION FUNCTIONS
    results = run_experiment(activation_fun_experiment(model_dict, train_dict),
                             out_dir='activation', test_data=test_data)
    plot_val_loss(results, dirname='activation')
    plot_val_accuracy(results, dirname='activation')

    # Experiment - OPTIMIZERS
    repeat = 2
    results = run_experiment(optimizer_experiment(model_dict, train_dict), out_dir='optimizer',
                             test_data=test_data)

    results['label'] = [res_label[10:] for res_label in results['label']]

    plot_val_loss(results, dirname='optimizer')
    plot_val_accuracy(results, dirname='optimizer')
    plot_time_bar(results, dirname='optimizer')
'''