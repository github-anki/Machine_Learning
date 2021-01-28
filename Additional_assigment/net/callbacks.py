import os
import re

import numpy as np

from settings import SAVE_PATH
from utils import ensure_dir_path_exists

BEST_MODEL_FILENAME = "best_model.pkl"
BEST_VAL_ACCURACY_FILENAME = "best_val_accuracy.txt"
DUMP_FILENAME = "model_dump.txt"


class Callback:
    def __init__(self):
        self.cb_name = camel2snake(type(self).__name__)
        self.trainer = None
        self.params = None

    def set_params(self, params):
        self.params = params

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def on_epoch_begin(self, **kwargs):
        pass

    def on_epoch_end(self, **kwargs):
        pass

    def on_batch_begin(self, **kwargs):
        pass

    def on_batch_end(self, **kwargs):
        pass

    def on_train_epoch_begin(self, **kwargs):
        pass

    def on_train_epoch_end(self, **kwargs):
        pass

    def on_predict_epoch_begin(self, **kwargs):
        pass

    def on_predict_epoch_end(self, **kwargs):
        pass


def camel2snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


class ModelDump(Callback):
    def __init__(self, output_dir):
        super().__init__()
        self._output_dir = output_dir
        self._save_dir = os.path.join(SAVE_PATH, self._output_dir)
        ensure_dir_path_exists(self._save_dir)

    def on_train_begin(self, **kwargs):
        self.trainer.model.dump(os.path.join(self._save_dir, DUMP_FILENAME))


class LoggerUpdater(Callback):
    def __init__(self):
        super().__init__()
        self._init()

    def _init(self):
        self._loss_accum = []
        self._train_preds = []
        self._y_shuffle_train = []

    def on_epoch_begin(self, **kwargs):
        self._init()

    def on_batch_end(self, **kwargs):
        x, y = kwargs.get('batch')
        epoch = kwargs.get('epoch')
        train_pred = kwargs.get('train_pred')
        train_loss = kwargs.get('train_loss')

        self._y_shuffle_train.extend(y)
        self._loss_accum.append(train_loss)
        self._train_preds.extend(train_pred)

        if epoch == 1:
            self.trainer.update_batch_log(train_loss)

    def on_train_epoch_end(self, **kwargs):
        epoch = kwargs.get('epoch')
        train_loss = np.mean(self._loss_accum)
        train_metrics = self.trainer.model.eval_metrics(self._train_preds, self._y_shuffle_train)
        train_acc = train_metrics['label_accuracy']

        val_metrics, val_loss = self.trainer.validate()
        val_acc = val_metrics['label_accuracy']

        self.trainer.update_epoch_log('train', train_loss, train_acc)
        self.trainer.update_epoch_log('val', val_loss, val_acc)

        print(
            "[epoch = %d] train_loss = %.5f, train_acc = %.3f,  val_loss = %.5f, val acc = %.3f\r" %
            (epoch, float(train_loss), train_acc, val_loss, val_acc), flush=True)


class SaveBestModel(Callback):
    """
    Saves best model params
    """

    def __init__(self, output_dir):
        super().__init__()
        self._output_dir = output_dir
        self._best_accuracy = 0
        self._save_dir = os.path.join(SAVE_PATH, self._output_dir)
        ensure_dir_path_exists(self._save_dir)
        self.save_path = os.path.join(self._save_dir, BEST_MODEL_FILENAME)

    def on_epoch_end(self, **kwargs):
        epoch = kwargs.get("epoch")

        last_val_accuracy = self.trainer.logger.logging_data['accuracy']['val'][-1]

        if last_val_accuracy > self._best_accuracy:
            self._best_accuracy = last_val_accuracy
            self.trainer.model.save_variables(self.save_path)
            self._write_accuracy(epoch)

    def _write_accuracy(self, epoch):
        """
        Writes and saves best achieved accuracy during training
        :param epoch: epoch number in which that accuracy occurred
        """
        with open(os.path.join(self._save_dir, BEST_VAL_ACCURACY_FILENAME), "w") as f:
            f.write("accuracy = %f\n" % self._best_accuracy)
            f.write("epoch = %d\n" % epoch)
