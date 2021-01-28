# Neural network
Neural network implemented in numpy. 

## Overview
Project written mainly for university classes. It includes implementation of dense, convolution, pooling, flatten and dropout layer, different optimizers,  weight initializers and train callbacks. For research purpose it allows to conduct some experiments to compare specific parameters of MLP and CONV nets. Project does not support GPU runtime.

## Installation guide

To run project you must have Anaconda installed. 
Clone the repository:
```
git clone https://github.com/Joanna065/neural-network.git
```
Create new conda environment for this project:
```
conda create -n <net-env> python=3.7
conda activate <net-env>
```
Install requirements listed in `requirements.txt` file via conda or pip:
```
while read requirement; do conda install --yes $requirement; done < requirements.txt
```
```
pip install -r requirements.txt
```
Project uses mnist dataset consisting of number images. Download data from github: 
`https://github.com/mnielsen/neural-networks-and-deep-learning.git`. 

Create file named `user_settings.py` in project root directory and save there absolute path for dataset `DATA_PATH` and directory for results of experiments `SAVE_PATH`. Example:
```
DATA_PATH = '/home/joanna/lab/neural_net/data/mnist.pkl'
SAVE_PATH = '/home/joanna/lab/neural_net/results/exp_mlp'
```
If necessary, add project directory to python from root dir localization:
```
export PYTHONPATH=`pwd`
```

## Usage
Running training is possible in two scripts:
* via terminal using ArgParser to set up training and model parameters - use script `experiments/train.py`

Example:
```
python experiments/train.py -m MlpNet -in Xavier -opt Adam --stop-epoch 10 'my_nets/simple_net'
```
More options to declare in ArgParser are visible in methof `parse_args()`.

* using script `experiments/run_training.py` where model and train parameters are explicitly declared in dicts in `__main__` function

Example:
```
out_dir = 'my_nets/simple_net'

model_dict = {
        'optimizer': SGD(),
        'initializer': Xavier(),
        'metrics': [LabelAccuracy()],
        'loss_fun': categorical_cross_entropy,
        'activation': Sigmoid,
        'hidden_units': (100,)
    }

    train_dict = {
        'train_data': train_data,
        'val_data': val_data,
        'epochs': 10,
        'batch_size': 50,
        'callbacks': [
            ModelDump(output_dir=out_dir),
            SaveBestModel(output_dir=out_dir),
            LoggerUpdater()
        ]
    }
```
Change values in dicts to run with different setup.

### Experiments
In `experiments` folder are prepared two scripts to conduct experiments applied to MLP and CONV nets. They focus on comparing different setups of training and model parameters and plot charts with time, loss and accuracy values. Feel free to change and write more experiments on your own. 
```
 # Experiment - initializers
    results = run_experiment(initializer_experiment(model_dict, train_dict), out_dir='initializer',
                             test_data=test_data)
    plot_val_loss(results, dirname='initializer')
    plot_val_accuracy(results, dirname='initializer')
```
Above example applies to experiment comparing weight initiliazer methods. Results (plots) are saved in declared earlier `SAVE_PATH` directory in newly created folder `initializer`.

## Reports
There are two reports (language Polish) included in this project. They contain analysis of experiment results and brief summary of conclusions.

`neural_nets_mlp.pdf` content:
1. Batch size influence on learning process
2. Impact of init weights range on training
3. Effect of the number of neurons in the hidden layer 
4. Comparison of sigmoid vs ReLU function
5. Comparison of different optimizers
6. Comparison of different weigh init methods
7. Effect of softmax vs MSE loss function

`neural_nets_conv_polish.pdf` content:
1. Effect of filter size on the classification results
2. Comparing CONV net with MLP net 


