import os
from collections import defaultdict
from time import time

from net.callbacks import LoggerUpdater, ModelDump, SaveBestModel
from net.initializers import *
from net.layers import ReLU
from net.losses import categorical_cross_entropy
from net.metrics import LabelAccuracy
from net.model import MlpNet, ConvNet
from net.optimizers import *
from settings import DATA_PATH
from training import Trainer
from utils import load_data, plot_val_loss, plot_val_accuracy, \
    plot_time_bar, save_results, open_results


def kernel_size_zero_pad_experiment(model_dict, train_dict):
    def exp_generator():
        for kernel_size in [3, 5, 7, 11]:
            model_dict['kernel_size'] = kernel_size
            yield model_dict, train_dict, 'kernel', str(kernel_size) + 'x' + str(kernel_size)

    return exp_generator


def kernel_size_same_pad_experiment(model_dict, train_dict, kernel_size, padding):
    def exp_generator():
        model_dict['kernel_size'] = kernel_size
        model_dict['padding'] = padding
        yield model_dict, train_dict, 'kernel', str(kernel_size) + 'x' + str(
            kernel_size) + ', pad=' + str(padding)

    return exp_generator


def calc_test_accuracy(model, test_data, train_dict):
    acc_metric = LabelAccuracy()
    x_test, y_test = test_data
    model.load_variables(train_dict['callbacks'][-1].save_path)
    accuracy = acc_metric(model.predict_classes(x_test), y_test)
    print('Accuracy on test data: {}'.format(accuracy))


def run_dropout_experiment(model_dict, train_dict, out_dir):
    np.random.seed(12345)
    results = defaultdict(list)
    model_dict['dropout'] = True
    kernel_size = model_dict['kernel_size']
    padding = model_dict['padding']
    label = f'kernel={kernel_size}x{kernel_size}, pad={padding}'

    model = ConvNet(**model_dict)

    train_dict['callbacks'][1] = ModelDump(output_dir=os.path.join(out_dir, label))
    train_dict['callbacks'][2] = SaveBestModel(output_dir=os.path.join(out_dir, label))
    trainer = Trainer(model, **train_dict)

    start_time = time()
    trainer.train_loop()
    time_period = (time() - start_time) / 60

    log_data = trainer.logger.logging_data

    results['model_dict'].append(model_dict)
    results['train_dict'].append(train_dict)
    results['time'].append(time_period)
    results['label'].append(label)
    results['log_data'].append(log_data)

    calc_test_accuracy(model, test_data, train_dict)

    save_results(out_dir, results_dict=results)
    return results


def run_mlp_conv_compare_experiment(model_dict_conv, model_dict_mlp, train_dict, out_dir,
                                    test_data):
    np.random.seed(12345)
    results = defaultdict(list)

    for model, model_dict in [(MlpNet(**model_dict_mlp), model_dict_mlp),
                              (ConvNet(**model_dict_conv), model_dict_conv)]:
        label = f'model={model.name}'
        print(f'{label}')
        train_dict['callbacks'][1] = ModelDump(output_dir=os.path.join(out_dir, label))
        train_dict['callbacks'][2] = SaveBestModel(output_dir=os.path.join(out_dir, label))
        trainer = Trainer(model, **train_dict)

        start_time = time()
        trainer.train_loop()
        time_period = (time() - start_time) / 60
        log_data = trainer.logger.logging_data

        results['model_dict'].append(model_dict)
        results['train_dict'].append(train_dict)
        results['time'].append(time_period)
        results['label'].append(label)
        results['log_data'].append(log_data)

        calc_test_accuracy(model, test_data, train_dict)

    save_results(out_dir, results_dict=results)
    return results


def run_experiment(experiment_generator, out_dir, test_data):
    np.random.seed(12345)
    results = defaultdict(list)

    for i, (model_dict, train_dict, exp_name, value) in enumerate(experiment_generator()):
        label = f'{exp_name}={value}'
        print(f'{i}. {label}')

        train_dict['callbacks'][1] = ModelDump(output_dir=os.path.join(out_dir, label))
        train_dict['callbacks'][2] = SaveBestModel(output_dir=os.path.join(out_dir, label))
        model = ConvNet(**model_dict)
        trainer = Trainer(model, **train_dict)

        start_time = time()
        trainer.train_loop()
        time_period = (time() - start_time) / 60

        log_data = trainer.logger.logging_data

        results['model_dict'].append(model_dict)
        results['train_dict'].append(train_dict)
        results['time'].append(time_period)
        results['label'].append(label)
        results['log_data'].append(log_data)

        calc_test_accuracy(model, test_data, train_dict)

    save_results(out_dir, results_dict=results)
    return results


if __name__ == "__main__":
    train_data, val_data, test_data = load_data(DATA_PATH)

    model_dict_conv = {
        'optimizer': Adam(),
        'initializer': KaimingHe(),
        'metrics': [LabelAccuracy()],
        'loss_fun': categorical_cross_entropy,
        'activation': ReLU,
        'kernel_size': 3,
        'filters': 8,
        'stride': 1,
        'padding': 0,
        'dropout': False
    }

    model_dict_mlp = {
        'optimizer': Adam(),
        'initializer': KaimingHe(),
        'metrics': [LabelAccuracy()],
        'loss_fun': categorical_cross_entropy,
        'activation': ReLU,
        'hidden_units': (100,)
    }

    train_dict = {
        'train_data': train_data,
        'val_data': val_data,
        'epochs': 25,
        'batch_size': 64,
        'callbacks': [LoggerUpdater(), None, None]
    }
    '''
    # Experiment - KERNEL SIZE, ZERO PADDING
    dir_name = 'kernel_size_zero_pad'
    results = run_experiment(kernel_size_zero_pad_experiment(model_dict_conv, train_dict),
                             out_dir=dir_name, test_data=test_data)
    plot_val_loss(results, dirname=dir_name)
    plot_val_accuracy(results, dirname=dir_name)
    plot_time_bar(results, dirname=dir_name, time_unit='min.')

    results = open_results(out_dir=dir_name)
    print(results)
    '''
    # Experiment - KERNEL SIZE, PADDING SAME

    # dir_name = 'kernel_3x3_same_pad'
    # results = run_experiment(kernel_size_same_pad_experiment(model_dict_conv, train_dict, 3, 1),
    #                         out_dir=dir_name, test_data=test_data)
    # plot_val_loss(results, dirname=dir_name)
    # plot_val_accuracy(results, dirname=dir_name)
    # plot_time_bar(results, dirname=dir_name, time_unit='min.')

    # dir_name = 'kernel_5x5_same_pad'
    # results = run_experiment(kernel_size_same_pad_experiment(model_dict_conv, train_dict, 5, 2),
    #                         out_dir=dir_name, test_data=test_data)
    # plot_val_loss(results, dirname=dir_name)
    # plot_val_accuracy(results, dirname=dir_name)
    # plot_time_bar(results, dirname=dir_name, time_unit='min.')

    # dir_name = 'kernel_7x7_same_pad'
    # results = run_experiment(kernel_size_same_pad_experiment(model_dict_conv, train_dict, 7, 3),
    #                         out_dir=dir_name, test_data=test_data)
    # plot_val_loss(results, dirname=dir_name)
    # plot_val_accuracy(results, dirname=dir_name)
    # plot_time_bar(results, dirname=dir_name, time_unit='min.')

    dir_name = 'kernel_11x11_same_pad'
    results = run_experiment(kernel_size_same_pad_experiment(model_dict_conv, train_dict, 11, 5),
                             out_dir=dir_name, test_data=test_data)
    plot_val_loss(results, dirname=dir_name)
    plot_val_accuracy(results, dirname=dir_name)
    plot_time_bar(results, dirname=dir_name, time_unit='min.')

    model_dict_conv = {
        'optimizer': Adam(),
        'initializer': KaimingHe(),
        'metrics': [LabelAccuracy()],
        'loss_fun': categorical_cross_entropy,
        'activation': ReLU,
        'kernel_size': 7,
        'filters': 8,
        'stride': 1,
        'padding': 3,  # 'same' padding
        'dropout': False
    }

    # Experiment - CONVOLUTION VS MLP COMPARE
    model_dict_mlp = {
        'optimizer': Adam(),
        'initializer': KaimingHe(),
        'metrics': [LabelAccuracy()],
        'loss_fun': categorical_cross_entropy,
        'activation': ReLU,
        'hidden_units': (100,)
    }
    '''
    dir_name = 'conv7x7_vs_mlp_adam'
    results = run_mlp_conv_compare_experiment(model_dict_conv, model_dict_mlp, train_dict,
                                              dir_name, test_data)
    plot_val_loss(results, dirname=dir_name)
    plot_val_accuracy(results, dirname=dir_name)
    plot_time_bar(results, dirname=dir_name, time_unit='min.')
    # Experiment - DROPOUT
    dir_name = 'conv7x7_dropout'
    results = run_dropout_experiment(model_dict_conv, train_dict, out_dir=dir_name)
    plot_val_loss(results, dirname=dir_name)
    plot_val_accuracy(results, dirname=dir_name)
    plot_time_bar(results, dirname=dir_name, time_unit='min.')
    '''
